import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
import shutil
import json
from contextlib import nullcontext
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import subprocess
import requests
import cv2

from rembg.session_factory import new_session
from rembg import remove as segmask_predict_fn

HfFolder.save_token(os.environ["HF_TOKEN"])

logger = get_logger(__name__)

def upload_file_to_s3(file: str, uri: str, concurrency: int = 10):
    from scaleml.utils.formats import parse_attachment_url
    from scaleml.data import storage_client
    from boto3.s3.transfer import TransferConfig

    r = parse_attachment_url(uri)
    bucket, key = r.bucket, r.key
    s3 = storage_client.sync_storage_client()

    # Enable multipart beyond threshold
    GB = 1024 ** 3
    config = TransferConfig(multipart_threshold=5 * GB, max_concurrency=concurrency)

    resp = s3.upload_file(Filename=file, Bucket=bucket, Key=key, Config=config)
    return resp

def download_folder_images(uri: str):
    from scaleml.utils.formats import parse_attachment_url
    from scaleml.data import storage_client

    print(f"Downloading images from s3 {uri}")
    r = parse_attachment_url(uri)
    bucket, key = r.bucket, r.key
    s3 = storage_client.sync_storage_client()
    if not os.path.exists(os.path.basename(key)):
        os.makedirs(os.path.basename(key))
    for obj in s3.list_objects(Bucket=bucket, Prefix=key)["Contents"]:
        base_dir_idx = obj["Key"].split(os.sep).index(os.path.basename(key))
        os.makedirs(os.path.join(*obj["Key"].split(os.sep)[base_dir_idx:-1]), exist_ok=True)
        s3.download_file(bucket, obj["Key"], os.path.join(*obj["Key"].split(os.sep)[base_dir_idx:]))
    os.system('ls -l "%s"' % os.path.basename(key))
    return os.path.basename(key)

def download_model_s3(model_name: str):
    import shutil
    import os
    from scaleml.utils.formats import parse_attachment_url
    from scaleml.data import storage_client
    
    s3 = storage_client.sync_storage_client()

    save_path = f"{model_name}"
    if os.path.exists(save_path):
        print(f"Found model already saved at {save_path}!")
        return save_path

    try:
        r = parse_attachment_url(model_name)
        bucket = r.bucket
        key = r.key
    except Exception as e:
        print(f"Could not parse model name as full s3 path! {e}")
        bucket = "scale-ml"
        key = f"catalog/gen/dreambooth/models/{model_name}.zip"
        print(f"Performing default lookup for model at {key} instead")
    print(f"Downloading model from {key}")
    s3.download_file(bucket, key, "model.zip")
    os.makedirs(save_path)
    print(f"Saving model to {save_path}")
    shutil.unpack_archive("model.zip", save_path, "zip")
    if len(os.listdir(save_path)) == 1:
        # zip needs to go in one deeper level
        save_path = f"{save_path}/{os.listdir(save_path)[0]}"
    print(f"Finished downloading! Files in path: {os.listdir(save_path)}")
    return save_path

def handle_instance_urls(urls, exp_name=None):
    if not exp_name:
        exp_name = int(time.time())
    urls = urls.split(' ')
    dirname = f"tmp_{exp_name}"
    os.makedirs(dirname)
    for url in urls:
        subprocess.call(f"wget {url}", shell=True, cwd=dirname)
    return dirname

def save_pretrained(exp_name: str, output_dir: str, pipeline: StableDiffusionInpaintPipeline, freeze_dir=None, ckpt_name='final'):
    print(f"Saving model to {output_dir}")
    pipeline.save_pretrained(output_dir)
    if freeze_dir and os.path.exists(freeze_dir):
           subprocess.call('mv -f '+freeze_dir +'/*.* '+ output_dir+'/text_encoder', shell=True)
           subprocess.call('rm -r '+ freeze_dir, shell=True)
    if exp_name is None:
        return
    zip_location = output_dir + '.zip'
    # TODO: make less horrible
    subprocess.call('cd %s; zip -r -0 %s .' % (output_dir, zip_location), shell=True)
    s3_path = f"s3://scale-ml/catalog/gen/inpainting/finetuned/{exp_name}/ckpt_{ckpt_name}.zip"
    print(f"Saving model to {s3_path}")
    upload_file_to_s3(zip_location, s3_path)
    print(f"Finished saving!")
    return s3_path

def push_bundle(exp_name: str, launch_env, ckpt_name='final', clone_bundle=None, custom_config={}):
    bundle_name = f"catalog-gen-fti-{exp_name}"
    endpoint_name = f"catalog-gen-fti-{exp_name}"
    model_name = f"{exp_name}/ckpt_{ckpt_name}"
    logger.info(
        f"Saving Launch endpoint with bundle {bundle_name} at endpoint {endpoint_name}, for model {model_name}"
    )
    try:
        from launch_internal import get_launch_client

        client = get_launch_client(
            api_key="catalog-ml",
            env=launch_env,
        )
    except Exception as e:
        logger.error(f"Could not get Launch client! {e}")

    try:
        existing_endpoint = client.get_model_endpoint(endpoint_name)
        if existing_endpoint is not None:
            logger.warning(f"Found existing endpoint {endpoint_name}! Removing...")
            client.delete_model_endpoint(endpoint_name)
    except:
        logger.info(f"Ignoring the non-existent endpoint {endpoint_name}")

    try:
        existing_bundle = client.get_model_bundle(bundle_name)
        logger.warning(f"Found existing bundle {bundle_name}! Removing...")
        client.delete_model_bundle(bundle_name)
    except:
        logger.info(f"Ignoring the non-existent bundle {bundle_name}")

    try:
        new_bundle = client.clone_model_bundle_with_changes(
            clone_bundle or "inpainting-1-03-06-627BAEE5",  # Copy current prod bundle
            bundle_name,
            {
                "model_type": "inpainting",
                "model_name": model_name,
            },
        )
    except Exception as e:
        print(f"Could not copy and recreate model bundle! {e}")

    ENDPOINT_CONFIG = {
        "min_workers": 1,
        "max_workers": 20,
        "per_worker": 1,
        "cpus": 7,
        "memory": "16Gi",
        "gpus": 1,
        "gpu_type": "nvidia-ampere-a10",
        "endpoint_type": "async",
        "labels": {"team": "catalog", "product": "forge-inpainting"},
    }
    ENDPOINT_CONFIG.update(custom_config)

    try:
        client.create_model_endpoint(
            model_bundle=bundle_name, endpoint_name=endpoint_name, **ENDPOINT_CONFIG
        )
        logger.info(f"Generating Launch endpoint succeeded!")
    except Exception as e:
        logger.error(f"Could not create new Launch endpoint! {e}")


def segmask_predict(image):
    transparent_layer = np.asarray(image.split()[-1])

    # For transparent background images, add a background
    background = Image.new("RGBA", image.size, (255, 255, 255))
    # print(background.size, image.size)
    image = Image.alpha_composite(background, image)

    mask = segmask_predict_fn(image, only_mask=True, session=rembg_session)
    mask = np.asarray(mask)
    mask_bool = mask != 0
    transparent_layer_bool = transparent_layer != 0
    mask = mask * (mask_bool & transparent_layer_bool).astype(mask.dtype)
    # mi = Image.fromarray(mask)
    d = random.randint(0, 100000)
    # Image.fromarray(mask).save('%dorig.png' % d)
    if _bordersize is not None:
        # pass
        mask = add_border(mask, _bordersize, region_range = _borderrange, remove_mask = _removemask)

    mask2 = add_border(mask, 24)
    border = mask2 - mask
    # border = (border / 255).round() + 0.00005
    border = (1 - border / 255).round() + 0.05
    border = border[None]
    # print(border[200])

    # Only keep parts of the mask that weren't originally transparent.
    mask = Image.fromarray(mask)
    # mi.save('m1.png')
    # mask.save('%dmutate.png' % d)
    return mask, torch.from_numpy(border)

def default_weights(mask):
    w = np.ones_like(mask)
    w = w[None]
    return torch.from_numpy(w)

def random_wedge(size, radius, width_degrees):
    # Create a new image
    image = Image.new("L", size, "black")
    draw = ImageDraw.Draw(image)
    # Generate a random starting angle for the wedge
    start_angle = random.uniform(0, 360)
    # Generate the end angle for the wedge
    end_angle = start_angle + width_degrees
    # Draw the wedge
    draw.pieslice([size[0]/2-radius, size[1]/2-radius, size[0]/2+radius, size[1]/2+radius], start_angle, end_angle, fill ='white',outline ='white')
    return np.asarray(image)

#random_wedge((300,300),150*1.4,90)

def add_border(mask: np.ndarray, n: int, threshold = 250, region_range = None, remove_mask = False):
    result = mask.copy()
    rows, cols = mask.shape
    if isinstance(n, tuple):
        n = random.randint(*n)
    if region_range is not None:
        deg = random.randint(*region_range)
        area = random_wedge((rows, cols), max(rows,cols)*1.42/2, deg)

    for i in range(rows):
        for j in range(cols):
            # If the current pixel is white
            if mask[i, j] >= threshold and (region_range is None or area[i, j] > 0):
                row_start = max(0, i - n)
                row_end = min(rows, i + n + 1)
                col_start = max(0, j - n)
                col_end = min(cols, j + n + 1)
                result[row_start:row_end, col_start:col_end] = 255

    if remove_mask:
        # zero out the original mask
        result[mask > 0] = 0
    # Return the result
    return result

_weightedloss = False
_rembgmasks = False
_bordersize = None#(16, 32)
_borderrange = None
# in degrees
# _borderrange = (10,180)
_removemask = False
_relightmodel = False
_relightprob = 1
_lightoutput = True
rembg_session = None
_invert = False
_nomask = False
_nomasksometimes = False
_fullmaskrate = 0
# actual rate is multiplied by fullmaskrate
_nomaskrate = 0#0.5
def get_bounds(mask):
    binarized_mask = np.asarray(mask.convert("RGBA")).copy()
    binarized_mask = np.mean(binarized_mask, axis=2).astype(np.uint8)
    binarized_mask = cv2.threshold(
        binarized_mask, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    contours, hierarchy = cv2.findContours(binarized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    print(len(contours[0]))

    # Find object
    obj_pixels = np.argwhere(binarized_mask == 255)
    obj_bbox = (
        obj_pixels[:, 0].min(),
        obj_pixels[:, 1].min(),
        obj_pixels[:, 0].max(),
        obj_pixels[:, 1].max(),
    )
    #xmin, ymin, xmax, ymax = obj_bbox
    return obj_bbox, contours

def valid_mask(mask):
    (xmin, ymin, xmax, ymax), contours = get_bounds(mask)
    width, height = xmax-xmin, ymax-ymin
    print('w', width)
    print('h', height)
    if width < 100 and height < 100:
        return '%s%s' % (width,height)
    if len(contours) > 1:
        return '%s' % len(contours)
    return True

def prepare_mask_and_masked_image(image, mask):
    # s = valid_mask(mask)
    # print(image.shape)
    if not isinstance(image, Image.Image):
        image = torch_to_image(image)

    orig = image
    if random.random() < _relightprob:
        image = transforms.ColorJitter(
            brightness=.7,#(.75, 1.1),
            contrast=.7,#(.85, 1.05),
            # saturation=.5,
            # hue=.5,
        )(image)
    # image = random_light(lightmodel, image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))

    # bordersz = random.randint(*_bordersize)
    # old_mask = mask
    # mask = add_border(mask, bordersz, threshold = 60, region_range = _borderrange, remove_mask = _removemask)
    # border_only = mask - old_mask

    # print(image.shape)
    # if random.random() < 0.5:
        # image[border_only > .8] = (255, 255, 255)

    if isinstance(image, np.ndarray):
        image = image[None].transpose(0, 3, 1, 2)[0]
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # print(image.shape)
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    # higher is less inclusive, lower is more inclusive
    threshold = 0.5
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    if _invert or _rembgmasks:
        mask = 1 - mask
    mask = torch.from_numpy(mask)
    # print(mask.shape)
    # print(image.shape)
#    mask = mask.reshape(image.shape)
    # print(image.shape)

    masked_image = image * (mask < 0.5)
    mi2 = image * (mask >= 0.5)
    # print(masked_image.shape)

    # mi = (masked_image / 2.0 + 0.5).clamp(0, 1)
    # nmi = mi.cpu().permute(0, 2, 3, 1).float().numpy()
    # if nmi.ndim == 3:
    #     nmi = nmi[None, ...]
    # nmi = (nmi * 255).round().astype("uint8")    
    # im = Image.fromarray(nmi[0])
    # im.save('im%d.png' % ri)
    # orig.save('orig%d.png' % ri)
    # ri = random.randint(0, 100000)
    # torch_to_image(masked_image).save('a%d.png' % ri)
    # orig.save('b%d.png' % ri)
    # torch_to_image(mi2).save('b%d.png' % ri)
    # if s != True:
    #     torch_to_image(masked_image).save('m%d_%s.png' % (ri, s))
    #     return None, None
    # torch_to_image(masked_image).save('m%d.png' % ri)

    return mask, masked_image

def torch_to_image(tensor):
    i = (tensor / 2.0 + 0.5).clamp(0, 1)
    if i.ndim == 3:
        i = i[None]
    ni = i.cpu().permute(0, 2, 3, 1).float().numpy()
    if ni.ndim == 3:
        ni = ni[None, ...]
    ni = (ni * 255).round().astype("uint8")    
    im = Image.fromarray(ni[0]).convert("RGBA")
    # print(im.size)
    # im.save('im.png')
    return im

def get_cutout_holes(height, width, min_holes=8, max_holes=32, min_height=16, max_height=128, min_width=16, max_width=128):
    holes = []
    for _n in range(random.randint(min_holes, max_holes)):
        hole_height = random.randint(min_height, max_height)
        hole_width = random.randint(min_width, max_width)
        y1 = random.randint(0, height - hole_height)
        x1 = random.randint(0, width - hole_width)
        y2 = y1 + hole_height
        x2 = x1 + hole_width
        holes.append((x1, y1, x2, y2))
    return holes

# generates random mask if fill is None, otherwise a mask filled with fill (int from 0 to 255)
def generate_mask(im_shape, fill=None):
    mask = torch.zeros(im_shape)
    if fill is not None:
        mask.fill_(fill)
    else:
        holes = get_cutout_holes(mask.shape[0], mask.shape[1])
        for (x1, y1, x2, y2) in holes:
            mask[y1:y2, x1:x2] = 255.
    # print(mask.shape)
    nparray = mask.cpu().float().numpy()
    im = Image.fromarray(nparray).convert("L")
    # print(im.size)
    # im.save('mask%d.png' % random.randint(0, 100000))
    return im, default_weights(nparray)

# generate random masks
def random_mask(im, ratio=1, mask_full_image=False):
    im_shape = im.shape[1:]
    if random.uniform(0, 1) < _fullmaskrate:
        colors = [0, 255]
        fillwhite = _nomask == _invert #(!_nomask && !_invert) || (_nomask && _invert)
        # do 0 mask some % of the time
        if random.uniform(0, 1) < _nomaskrate:
            fillwhite = _invert
#        mask.fill_(255.)
        return generate_mask(im_shape, fill=colors[fillwhite])
    if _rembgmasks:
        i = segmask_predict(torch_to_image(im))
        return i
    return generate_mask(im_shape)

    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # nate: 25% of time, use a full mask
    if random.uniform(0, 1) < 0.25:
        mask_full_image = True
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
        if _invert:
            # already all black so return directly
            return mask
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask

if _relightmodel:
    import sys
    sys.path.append('/home/ubuntu/Deep-Illuminator/app')
    import yaml
    import torch
    from probe_relighting.utils.preprocessing import denorm, open_image, open_probe
    from probe_relighting.network import ProbeRelighting
    from torchvision.transforms.functional import to_pil_image
    from pathlib import Path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FILE_PATH = Path('/home/ubuntu/Deep-Illuminator/app/probe_relighting/utils/demotools.py')
    lightmodel = get_model().to(device)

def get_model():
    experiment_path = FILE_PATH.parent / '../network_config.yaml'
    with open(experiment_path) as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    model = ProbeRelighting
    model = model(opt['model_opt'])
    model_path = FILE_PATH.parent / '../checkpoint.ckpt'
    model.load_state_dict(torch.load(model_path,
                                     map_location=device)['model_state_dict'])
    return model

def make_sample(img, style, idx):
    if style == 'synthetic':
        chrome_name = FILE_PATH.parent / f'../data/point_1kW/chrome_{str(idx).zfill(4)}.png'
        gray_name = FILE_PATH.parent / f'../data/point_1kW/gray_{str(idx).zfill(4)}.png'
    elif style == 'mid':
        chrome_name = FILE_PATH.parent / f'../data/mid_probes/dir_{idx}_chrome256.jpg'
        gray_name = FILE_PATH.parent / f'../data/mid_probes/dir_{idx}_gray256.jpg'
    chrome = open_probe(chrome_name).unsqueeze(0)
    gray = open_probe(gray_name).unsqueeze(0)
    return {'original': img, 'probe_1': chrome, 'probe_2': gray}

def get_output(model, img, idx, style='synthetic'):
    with torch.no_grad():
        model.eval()
        sample = make_sample(img, style, idx)
        probes = torch.cat((sample['probe_1'], sample['probe_2']), 3)
        output = model(sample)
        output = torch.cat([output['generated_img']], dim=2)
        output = denorm(output.cpu().squeeze())
        output = to_pil_image(output)
        return output

def random_light(model, img):
    # no lighting change 50% of the time
    if random.random() < 0.5:
        return img
    img = open_image(img).unsqueeze(0)
    styles = {
        'mid': (1, 24),
        'synthetic': (1, 360),
    }
    style = random.choice(list(styles.keys()))
    idx = random.randint(*styles[style])
    return get_output(model, img, idx, style)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--launch",
        type=int,
        default=-1,
        help="What checkpoint to launch an endpoint at. Defaults to no checkpoint launched.",
    )
    parser.add_argument(
        "--launch_env",
        type=str,
        default="staging",
        help="What environment to launch endpoint to.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of experiment",
    )
    parser.add_argument(
        "--instance_urls",
        type=str,
        help="List of instance URLs to train on, instead of an instance_dir"
    )
    parser.add_argument(
        "--clone_bundle",
        type=str,
        help="Name of bundle to clone for launch endpoint",
        default=None,
    )
    parser.add_argument(
        "--custom_config",
        type=str,
        help="JSON of a custom endpoint config to use. Will update the existing config with those values",
        default='{}'
    )
    parser.add_argument(
        "--rembg",
        default=False,
        action="store_true",
        help="whether to use rembg for generating masks (insead of random)",
    )
    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=1000000,
        help=("The step at which the text_encoder is no longer trained"),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None and args.instance_urls is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    if args.rembg:
        global _invert
        global _rembgmasks
        global rembg_session
        _rembgmasks = True
        # TODO: do this less hackily
        _invert = _invert or _rembgmasks
        rembg_session = new_session("u2net")
    if args.pretrained_model_name_or_path and args.pretrained_model_name_or_path.startswith('s3'):
        args.pretrained_model_name_or_path = download_model_s3(args.pretrained_model_name_or_path)

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list([x for x in Path(instance_data_root).iterdir() if not x.is_dir()])
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["PIL_images"] = instance_image
        example["instance_images"] = self.image_transforms(instance_image)
        mask, weights = random_mask(example["instance_images"], 1, False)
        example["instance_masks"], example["instance_masked_images"] = prepare_mask_and_masked_image(example["instance_images"], mask)
        if example['instance_masks'] is None:
            print('bad', self.instance_images_path[index % self.num_instance_images])
        example["weights"] = weights

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_PIL_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def main():
    args = parse_args()
    if args.output_dir is None:
        print(f"Setting output dir to {args.exp_name}/")
        args.output_dir = args.exp_name

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype, safety_checker=None
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size, num_workers=1
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)
            transform_to_pil = transforms.ToPILImage()
            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                bsz = len(example["prompt"])
                fake_images = torch.rand((3, args.resolution, args.resolution))
                transform_to_pil = transforms.ToPILImage()
                fake_pil_images = transform_to_pil(fake_images)

                fake_mask = random_mask((args.resolution, args.resolution), ratio=1, mask_full_image=True)

                images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #             args.pretrained_model_name_or_path, safety_checker=None, #torch_dtype=torch.float16
    # )
    # out = pipe('', Image.open('/home/ubuntu/bottle.png'), Image.open('/home/ubuntu/bottle_mask.png'))
    # print(out)
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                               # low_cpu_mem_usage=False, ignore_mismatched_sizes=False
                                               )
    print(unet.conv_out.weight.shape)
    unet.conv_out.weight = torch.nn.Parameter(torch.cat([unet.conv_out.weight, 
                                                         unet.conv_out.weight,
                                                         # torch.zeros_like(unet.conv_out.weight),
                                                         torch.zeros((1, 320, 3, 3))+.1
                                                        ]))
    print(unet.conv_out.weight.shape)
    print(unet.conv_out.bias.shape)
    unet.conv_out.bias = torch.nn.Parameter(torch.cat([unet.conv_out.bias,
                                                       unet.conv_out.bias,
                                                       # torch.zeros_like(unet.conv_out.bias),
                                                       torch.zeros((1,)) + .1
                                                      ]))

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    # noise_scheduler = DDPMScheduler(
    #     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    # )

    if args.instance_urls:
        if args.instance_data_dir:
            print('Warning: received both --instance_urls and --instance_data_dir, defaulting to use the --instance_urls')
        args.instance_data_dir = handle_instance_urls(args.instance_urls, args.exp_name)

    if args.instance_data_dir.startswith("s3://"):
        args.instance_data_dir = download_folder_images(args.instance_data_dir)

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            ]
        )
        batchlen = len(examples)
        examples = [example for example in examples if example['instance_masks'] is not None]
        if len(examples) < batchlen:
            db_len = len(train_dataset)
            diff = batchlen - len(examples)
            while diff != 0:
                a = train_dataset[np.random.randint(0, db_len)]
                if a['instance_masks'] is None:
                    continue
                examples.append(a)
                diff -= 1

        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            pior_pil = [example["class_PIL_images"] for example in examples]

        masks = [example["instance_masks"] for example in examples]
        weights = [example["weights"] for example in examples]
        masked_images = [example["instance_masked_images"] for example in examples]
#         for example in examples:
#             #pil_image = example["PIL_images"]
#             pil_image = example["instance_images"]
#             # generate a random mask
#             mask = random_mask(pil_image.size, 1, False)
#             # apply transforms
#             #mask = image_transforms(mask)
#             #pil_image = image_transforms(pil_image)
#             # prepare mask and masked image
#             mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

#             masks.append(mask)
#             masked_images.append(masked_image)

        if args.with_prior_preservation:
            for pil_image in pior_pil:
                # generate a random mask
                mask = random_mask(pil_image.size, 1, False)
                # apply transforms
                mask = image_transforms(mask)
                pil_image = image_transforms(pil_image)
                # prepare mask and masked image
                mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

                masks.append(mask)
                masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        masks = torch.stack(masks)
        weights = torch.stack(weights)
        masked_images = torch.stack(masked_images)
        batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images, "weights": weights}
        return batch
    
    def get_t0(model_output, t, sample):
        # num_inference_steps = self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
        prev_t = -1

        if model_output.shape[1] == sample.shape[1] * 2 and noise_scheduler.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = noise_scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if noise_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif noise_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif noise_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if noise_scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        #                         pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        #                         current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        #                         pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        return pred_original_sample


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space

                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Convert masked images to latent space
                    # print('shapes', batch["pixel_values"].shape, batch["masked_images"].shape)
                    masked_latents = vae.encode(
                        batch["masked_images"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * 0.18215

                    masks = batch["masks"]
                    # print(masks.shape)
                    mask = F.interpolate(masks, scale_factor=1 / 8)
                    # vaemask = vae.encode(mask.to(dtype=weight_dtype)).latent_dist.sample()
                    weights = batch["weights"]
                    weight = F.interpolate(weights, scale_factor=1 / 8)
                    # print(weight.shape)
                    # print(weight[0][0][25])
                    # print(mask.shape)
                    # # resize the mask to latents shape as we concatenate the mask to the latents
                    # mask = torch.stack(
                    #     [
                    #         torch.nn.functional.interpolate(mask, size=(args.resolution // 8, args.resolution // 8))
                    #         for mask in masks
                    #     ]
                    # )
                    # mask = mask.reshape(-1, 1, args.resolution // 8, args.resolution // 8)
                    # print(mask.shape)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # concatenate the noised latents with the mask and the masked latents
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Get the text embedding for conditioning
                with text_enc_context:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                target = noise
                if hasattr(noise_scheduler.config, 'prediction_type'):
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    if _weightedloss:
                        sqs = (noise_pred.float() - target.float()) ** 2
                        loss = torch.sum(sqs * weight) / torch.sum(weight) / 4
                    elif _lightoutput:
                        print(noise_pred.shape)
                        noise_pred, lighting, alpha = torch.split(noise_pred, [4, 4, 1], dim=1)
                        print(noise_pred.shape, lighting.shape, alpha.shape)
                        # relit = noise_pred * .5 + (lighting * mask) * .5
                        # loss = F.mse_loss(relit.float(), target.float(), reduction="mean")

                        denoised = get_t0(noise_pred, timesteps, noisy_latents)
                        denoised_lighting = get_t0(lighting, timesteps, noisy_latents)
                        mi = torch.min(alpha)
                        mx= torch.max(alpha)
                        print('minmax', mi,mx)
                        alpha = (alpha-mi)/(mx-mi)
                        print('test', torch.min(alpha),torch.max(alpha))
                        # denoised_lighting = lighting

                        relit = (mask >= 0.5) * denoised + (mask < 0.5) * (denoised * (1-alpha) + denoised_lighting * alpha)
                        unlit_mask = (mask < 0.5) * denoised
                        masked_latents_mask = (mask < 0.5) * masked_latents
                        #im = vae.decode(1 / 0.18215 * denoised).sample
#                         with torch.no_grad():
#                             orig_im = vae.decode()
#                             do_transforms()
#                             orig_latents = vae.encode()

                        lit_loss = F.mse_loss(relit.float(), latents.float(), reduction="mean")
                        unlit_loss = F.mse_loss(unlit_mask.float(), masked_latents_mask.float(), reduction="mean")
                        loss = .5 * (lit_loss + unlit_loss)
                        print(lit_loss)
                        print(unlit_loss)
                        with torch.no_grad():
                            print(alpha.shape)
                            print(denoised.shape)
                            alpha_up = F.interpolate(alpha, scale_factor=8)
                            alpha_mask = torch.cat([alpha_up, alpha_up, alpha_up], dim=1)
                            print(alpha_mask.shape)
                            ri = random.randint(0,10000)
                            torch_to_image(alpha_mask.detach()).save('mask%d.png' % ri)
                            light_add = relit - denoised
                            light_add = (light_add - torch.min(light_add)) / (torch.max(light_add)-torch.min(light_add))
                            torch_to_image(light_add.detach()).save('light_add%d.png' % ri)

                            denoised_img = vae.decode(1 / 0.18215 * denoised).sample
                            torch_to_image(denoised_img.detach()).save('out%d.png' % ri)

                            im = vae.decode(1 / 0.18215 * denoised_lighting).sample
                            torch_to_image(im.detach()).save('outlight%d.png' % ri)
                            
                            orig = batch["masked_images"]
                            relit_final = (masks >= 0.5) * denoised_img + (masks < 0.5) * (orig * (1-alpha_up) + im * alpha_up)
                            torch_to_image(relit_final.detach()).save('final%d.png' % ri)
                            torch_to_image(orig.detach()).save('mi_full%d.png' % ri)
                            torch_to_image(batch['pixel_values'].detach()).save('true_orig%d.png' % ri)

                            im = vae.decode(1 / 0.18215 * unlit_mask).sample
                            torch_to_image(im.detach()).save('unlit%d.png' % ri)
                            im = vae.decode(1 / 0.18215 * masked_latents_mask).sample
                            torch_to_image(im.detach()).save('mi%d.png' % ri)

                            im = vae.decode(1 / 0.18215 * relit).sample
                            torch_to_image(im.detach()).save('lit%d.png' % ri)
                            im = vae.decode(1 / 0.18215 * latents).sample
                            torch_to_image(im.detach()).save('orig%d.png' % ri)
                        # denoised = denoised * mask
                        # loss += F.mse_loss(denoised.float(), masked_latents.float(), reduction="mean")
                    else:
                        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                    # sqs = (noise_pred.float() - target.float()) ** 2
                    # print(sqs.shape)
                    # print(weight.shape)
                    # print(loss, torch.mean(sqs), torch.sum(sqs))
                    # TODO: why is mean vs sum/n off by a factor of 4????
                    # loss = torch.sum(sqs * weight) / torch.sum(weight) / 4
                    # print(loss)
                    # print(torch.mean(sqs))
                    # print(torch.sum(sqs))
                    # print(torch.sum(sqs*weight))
                    # print(torch.sum(weight))
                    #print(torch.mean(sqs))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if args.train_text_encoder and global_step == args.stop_text_encoder_training:
                if accelerator.is_main_process:
                    print("\nFreezing the text_encoder...")
                    tmp_dir=args.output_dir+'/tmp_ckpt'
                    frz_dir=args.output_dir + "/text_encoder_frozen"
                    if os.path.exists(tmp_dir):
                        subprocess.call('rm -r '+ tmp_dir, shell=True)
                    os.mkdir(tmp_dir)
                    if os.path.exists(frz_dir):
                        subprocess.call('rm -r '+ frz_dir, shell=True)
                    os.mkdir(frz_dir)
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                    )
                    pipe.save_pretrained(tmp_dir)
                    subprocess.call('mv ' + tmp_dir + "/text_encoder/*.* " + frz_dir, shell=True)
                    subprocess.call('rm -r '+ tmp_dir, shell=True)
                    del pipe

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        frz_dir=args.output_dir + "/text_encoder_frozen" if args.train_text_encoder else None

#        output_dir = os.path.join(args.output_dir, "final")
#        if not os.path.exists(output_dir):
#            os.makedirs(output_dir)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        try:
            shutil.copy(sys.argv[0], args.output_dir)
        except Exception as e:
            print('error', e)

        save_pretrained(args.exp_name, args.output_dir, pipeline, freeze_dir=frz_dir)

        if args.exp_name is not None:
            push_bundle(args.exp_name, args.launch_env, clone_bundle=args.clone_bundle, custom_config=json.loads(args.custom_config))

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
