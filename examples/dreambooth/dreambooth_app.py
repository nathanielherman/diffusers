# ---
# deploy: true
# integration-test: false
# ---
#
# # Pet Art Dreambooth with Hugging Face and Gradio
#
# This example finetunes the [Stable Diffusion v1.5 model](https://huggingface.co/runwayml/stable-diffusion-v1-5)
# on images of a pet (by default, a puppy named Qwerty)
# using a technique called textual inversion from [the "Dreambooth" paper](https://dreambooth.github.io/).
# Effectively, it teaches a general image generation model a new "proper noun",
# allowing for the personalized generation of art and photos.
# It then makes the model shareable with others using the [Gradio.app](https://gradio.app/)
# web interface framework.
#
# It demonstrates a simple, productive, and cost-effective pathway
# to building on large pretrained models
# by using Modal's building blocks, like
# [GPU-accelerated](https://modal.com/docs/guide/gpu#using-a100-gpus-alpha) Modal Functions, [shared volumes](https://modal.com/docs/guide/shared-volumes#shared-volumes) for caching, and [Modal webhooks](https://modal.com/docs/guide/webhooks#webhook).
#
# And with some light customization, you can use it to generate images of your pet!
#
# ![Gradio.app image generation interface](./gradio-image-generate.png)
#
# ## Setting up the dependencies
#
# We can start from a base image and specify all of our dependencies.

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

import modal

web_app = FastAPI()
assets_path = Path(__file__).parent / "catalog_data"
script_path = Path(__file__).parent / "011323_weightedloss_train_dreambooth_inpaint.py"
stub = modal.Stub(name="example-dreambooth-app")

# Commit in `diffusers` to checkout `train_dreambooth.py` from.
GIT_SHA = "ed616bd8a8740927770eebe017aedb6204c6105f"

image = (
    modal.Image.conda(python_version='3.10')
        .conda_install(
            "cudatoolkit=11.2",
            "cudnn=8.1.0",
            "cuda-nvcc",
            channels=["conda-forge", "nvidia"],
        )
    .pip_install(
        "accelerate",
        "datasets",
        "ftfy",
        "gradio~=3.10",
        "smart_open",
        "transformers",
        "torch",
        "torchvision",
        "triton",
        "rembg[gpu]",
        "bitsandbytes",
    )
    .pip_install("xformers", pre=True)
    .apt_install(
        "git",
    )
    .run_commands(
        f"cd /root && git clone https://github.com/huggingface/diffusers && cd diffusers && git checkout {GIT_SHA} && pip install -e ."
    )
    .apt_install(
        "zip",
    )
)

# A persistent shared volume will store model artefacts across Modal app runs.
# This is crucial as finetuning runs are separate from the Gradio app we run as a webhook.

volume = modal.SharedVolume().persist("dreambooth-finetuning-vol")
MODEL_DIR = Path("/model")

# ## Config
#
# All configs get their own dataclasses to avoid scattering special/magic values throughout code.
# You can read more about how the values in `TrainConfig` are chosen and adjusted [in this blog post on Hugging Face](https://huggingface.co/blog/dreambooth).
# To run training on images of your own pet, upload the images to separate URLs and edit the contents of the file at `TrainConfig.instance_example_urls_file` to point to them.


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "Qwerty"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "Golden Retriever"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for plaintext file with urls for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "dreambooth_app/instance_example_urls.txt"
    )

    # identifier for pretrained model on Hugging Face
    model_name: str = "runwayml/stable-diffusion-inpainting"

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1500


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5


# ## Get finetuning dataset
#
# Part of the magic of Dreambooth is that we only need 4-10 images for finetuning.
# So we can fetch just a few images, stored on consumer platforms like Imgur or Google Drive
# -- no need for expensive data collection or data engineering.

IMG_PATH = Path("/img")


def load_images(image_urls):
    import PIL.Image
    from smart_open import open

    os.makedirs(IMG_PATH, exist_ok=True)
    for ii, url in enumerate(image_urls):
        with open(url, "rb") as f:
            image = PIL.Image.open(f)
            image.save(IMG_PATH / f"{ii}.png")
    print("Images loaded.")

    return IMG_PATH


# ## Finetuning a text-to-image model
#
# This model is trained to do a sort of "reverse [ekphrasis](https://en.wikipedia.org/wiki/Ekphrasis)":
# it attempts to recreate a visual work of art or image from only its description.
#
# We can use a trained model to synthesize wholly new images
# by combining the concepts it has learned from the training data.
#
# We use a pretrained model, version 1.5 of the Stable Diffusion model. In this example, we "finetune" SD v1.5, making only small adjustments to the weights,
# in order to just teach it a new word: the name of our pet.
#
# The result is a model that can generate novel images of our pet:
# as an astronaut in space, as painted by Van Gogh or Bastiat, etc.
#
# ### Finetuning with Hugging Face 🧨 Diffusers and Accelerate
#
# The model weights, libraries, and training script are all provided by [🤗 Hugging Face](https://huggingface.co).
#
# To access the model weights, you'll need a [Hugging Face account](https://huggingface.co/join)
# and from that account you'll need to accept the model license [here](https://huggingface.co/runwayml/stable-diffusion-v1-5).
#
# Lastly, you'll need to create a token from that account and share it with Modal
# under the name `"huggingface"`. Follow the instructions [here](https://modal.com/secrets).
#
# Then, you can kick off a training job with the command
# `modal run dreambooth_app.py::stub.train`.
# It should take about ten minutes.
#
# Tip: if the results you're seeing don't match the prompt too well, and instead produce an image of your subject again, the model has likely overfit. In this case, repeat training with a lower # of max_train_steps. On the other hand, if the results don't look like your subject, you might need to increase # of max_train_steps.


@stub.function(
    image=image,
    gpu="A100",  # finetuning is VRAM hungry, so this should be an A100
    shared_volumes={
        str(
            MODEL_DIR
        ): volume,  # fine-tuned model will be stored at `MODEL_DIR`
    },
    timeout=9200,  # 30 minutes
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    mounts=[
        modal.Mount.from_local_dir(assets_path, remote_path="/data"),
        modal.Mount.from_local_file(script_path, remote_path="/root/train_dreambooth_inpaint.py")
    ],
)
def train(instance_example_urls, config=TrainConfig(), model_name="directlight2"):
    import subprocess

    import huggingface_hub
    from accelerate.utils import write_basic_config
    from transformers import CLIPTokenizer

    trained_dir = MODEL_DIR / model_name / "trained"
    pics_dir = MODEL_DIR / model_name / "pics"

    # set up runner-local image and shared model weight directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(trained_dir, exist_ok=True)
    subprocess.call('rm -rf %s' % str(pics_dir), shell=True)
    os.makedirs(pics_dir, exist_ok=True)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # authenticate to hugging face so we can download the model weights
    hf_key = os.environ["HF_TOKEN"]
    os.environ["HF_HOME"] = str(MODEL_DIR)
    huggingface_hub.login(hf_key)

    # define the training prompt
    instance_phrase = f"{config.instance_name} {config.class_name}"
    prompt = "phtrlsks"
    
    img_path = '/data'

    # run training -- see huggingface accelerate docs for details
    subprocess.run(
        [
            "accelerate",
            "launch",
            "/root/train_dreambooth_inpaint.py",
            "--train_text_encoder",  # needs at least 16GB of GPU RAM.
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={img_path}",
            f"--output_dir={trained_dir}",
            f"--instance_prompt='{prompt}'",
            f"--resolution={config.resolution}",
            f"--use_8bit_adam",
#            f"--stop_text_encoder_training=1",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--seed=1337",
            f"--rembg",
        ],
        check=True,
        cwd=pics_dir,
    )

    #subprocess.call('cd %s; zip -r -0 %s .' % (trained_dir, 'model.zip'), shell=True)


# ## Wrap the trained model in Gradio's web UI
#
# Gradio.app makes it super easy to expose a model's functionality
# in an easy-to-use, responsive web interface.
#
# This model is a text-to-image generator,
# so we set up an interface that includes a user-entry text box
# and a frame for displaying images.
#
# We also provide some example text inputs to help
# guide users and to kick-start their creative juices.
#
# You can deploy the app on Modal forever with the command
# `modal deploy dreambooth_app.py`.


@stub.asgi(
    image=image,
    gpu="A100",
    shared_volumes={str(MODEL_DIR): volume},
    mounts=[
        modal.Mount.from_local_dir(assets_path, remote_path="/data"),
        modal.Mount.from_local_file(script_path, remote_path="/root/train_dreambooth_inpaint.py")
    ],
)
def fastapi_app2(config=AppConfig()):
    import gradio as gr
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from gradio.routes import mount_gradio_app

    # set up a hugging face inference pipeline using our model
    ddim = DDIMScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR, scheduler=ddim, torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    # wrap inference in a text-to-image function
    def go(text):
        image = pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        return image

    instance_phrase = f"{config.instance_name} the {config.class_name}"

    example_prompts = [
        f"{instance_phrase}",
        f"a painting of {instance_phrase.title()} With A Pearl Earring, by Vermeer",
        f"oil painting of {instance_phrase} flying through space as an astronaut",
        f"a painting of {instance_phrase} in cyberpunk city. character design by cory loftis. volumetric light, detailed, rendered in octane",
        f"drawing of {instance_phrase} high quality, cartoon, path traced, by studio ghibli and don bluth",
    ]

    modal_docs_url = "https://modal.com/docs/guide"
    modal_example_url = f"{modal_docs_url}/ex/dreambooth_app"

    description = f"""Describe what they are doing or how a particular artist or style would depict them. Be fantastical! Try the examples below for inspiration.

### Learn how to make your own [here]({modal_example_url}).
    """

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        title=f"Generate images of {instance_phrase}.",
        description=description,
        examples=example_prompts,
        css="/assets/index.css",
        allow_flagging="never",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# ## Running this on the command line
#
# You can use the `modal` command-line interface to interact with this code,
# in particular training the model and running the interactive Gradio service
#
# - `modal run dreambooth_app.py` will train the model
# - `modal serve dreambooth_app.py` will [serve](https://modal.com/docs/guide/webhooks#developing-with-modal-serve) the Gradio interface at a temporarily location.
# - `modal shell dreambooth_app.py` is a convenient helper to open a bash [shell](https://modal.com/docs/guide/developing-debugging#stubinteractive_shell) in our image (for debugging)
#
# Remember, once you've trained your own fine-tuned model, you can deploy it using `modal deploy dreambooth_app.py`.
#
# This app is already deployed on Modal and you can try it out at https://modal-labs-example-dreambooth-app-fastapi-app.modal.run


@stub.local_entrypoint
def run():
    train.call([])
