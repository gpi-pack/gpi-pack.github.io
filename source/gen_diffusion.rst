.. _generate_images:
Generating Images without Diffusion Models
===========


If you want to use GPI for image data, you need to generate images and extract the internal representations of deep generative models. This section describes how to generate images and extract the internal representations using `Stable Diffusion models <https://huggingface.co/stabilityai>`_.

.. note::
    For data generation, we recommend users to use GPUs. See :ref:`gpu_usage_section`.


What is diffusion model?
---------

Diffusion models are a class of generative models that learn to generate data by modeling the diffusion process. They work by (1) gradually adding noise to the data and then (2) learning to reverse this process to generate new samples. This approach has been shown to produce high-quality samples in various domains, including images, videos, and audio. Below, we briefly give an overview of how one type of diffusion model, Stable Diffusion, works.

Stable diffusion model is a deep generative models developed by `stability.ai <https://stability.ai/>`_ and it is a type of diffusion model that uses a latent variable approach to generate high-quality images. It works by first encoding the input image into a lower-dimensional latent space, where the diffusion process is applied. The model then learns to reverse the diffusion process to generate new images from the latent space.

The stable diffusion model consists of the following key components:

1. **Autoencoder (VAE)**: The variational autoencoder (VAE) is used to both encode the input image into a low-dimensional latent space and decode the latent representation back into an image. For example, we can convert an image of size (3, 512, 512) (3 color channels, 512 pixels height, 512 pixels width) into a latent representation of size (4, 64, 64) (4 channels, 64 pixels height, 64 pixels width).

2. **UNet**: The U-Net architecture is used to predict the denoised image representation from the noisy latents.

3. **Text Encoder**: The text encoder is used to enable users to condition the image generation process on text prompts. It encodes the text input into a latent representation that can be used to guide the image generation process.

4. **Scheduler**: The scheduler is used to control the diffusion process, including the noise schedule and the number of diffusion steps. During the training phase, the scheduler adds noise to a sample to train a diffusion model. During the inference phase, the scheduler defines how to update a sample based on a pre-trained model's output.

In the context of GPI, we can use Stable Diffusion to generate images and extract the internal representations of the diffusion model. Since the internal representations within UNet is stochastic, we need to use the final output of the UNet (which is the input to the decoder) as the internal representation.


How to use Stable Diffusion
---------

The ``extract_images`` function provides a simple interface for generating images and extracting internal representations from Stable Diffusion models. This function is built on top of the Hugging Face `diffusers <https://huggingface.co/docs/diffusers/index>`_  library and automates the process of image generation and internal representation extraction.


.. note::
    This functionality is added in version 0.1.1, and we currently only support the previous version of Stable Diffusion models (version 1.5 and version 2.1). We will support the latest version of these models in the future releases.


Here is an example of how to use this function:

.. code-block:: python

    from gpi_pack.diffusion import extract_images

    # Load toy images (from the website)
    from PIL import Image
    from io import BytesIO

    # Generate images using Stable Diffusion and save the hidden states
    extract_images(
        images = img, # List of Input images or Images. If List is provided, the function will generate images based on each input images.
        prompts = "no change", # Text prompts to condition the image generation process.
        output_hidden_dir= "/content/save", # Directory to save the internal representations of the diffusion model.
        output_image_dir= "/content/save", # Directory to save the generated images.
        # Optional parameters
        save_name = "image", # Prefix for the saved image files
        prefix_hidden = "hidden_", # Prefix for the saved internal representation files
        model_id = "runwayml/stable-diffusion-v1-5", # Model ID of the Stable Diffusion model to use. You can specify any model from Hugging Face's diffusers library.
        strength = 0, # Strength of the image generation process (0.0 to 1.0). A higher value means more change from the input image. 0 means no change.
        guidance_scale = 7.5, # Guidance scale for the text-to-image generation. Higher values lead to more adherence to the text prompt.
        num_inference_steps = 50, # Number of diffusion steps to use for image generation
    )


Arguments
---------

The function ``extract_images`` has the following arguments:

- ``images``: input image(s) to transform. Accepts a single PIL.Image object, a single file path, or a list combining either type (required).

- ``prompts``: text prompt(s) paired one‑to‑one with images. Accepts a single string or a list of strings (required).

- ``output_hidden_dir``: directory where extracted hidden‑state tensors (latents) will be saved (required).

- ``output_image_dir``: directory to save the generated (or reconstructed) images. If omitted, images are not written to disk (optional).

- ``save_name``: base filename stem used when saving images (e.g., gen_0.png). Default is "gen".

- ``prefix_hidden``: prefix for hidden‑state files (e.g., hidden_0.pt). Default is "hidden".

- ``model_id``: Hugging Face identifier or local path of the Stable Diffusion model checkpoint to load. Default is "runwayml/stable-diffusion-v1-5".

- ``device``: compute device for inference (e.g., "cuda", "cpu"). If None, chooses "cuda" when available, else "cpu" (optional).

- ``cache_dir``: local directory for caching model weights. If None, uses Hugging Face’s default cache location (optional).

- ``strength``: how strongly the input image is altered (0 = no change, 1 = full generation). Float between 0 and 1. Default is 0.

- ``num_inference_steps``: number of diffusion denoising steps. Higher values yield higher quality but slower generation. Default is 50.

- ``guidance_scale``: classifier‑free guidance scale controlling prompt adherence. Typical range 5 – 15. Default is 7.5.

- ``negative_prompt``: text that describes features to avoid in the generated images. If None, no negative prompt is used (optional).

- ``seed``: integer random seed for deterministic results. If None, the entire generation is nondeterministic (optional).