.. _ref_StableDiffusionImg2ImgExtractor:

StableDiffusionImg2ImgExtractor
===============================

Description
-----------

The ``StableDiffusionImg2ImgExtractor`` class loads the VAE, CLIP tokenizer, text encoder, UNet, and DDIM scheduler used by a Stable Diffusion image-to-image checkpoint. It provides lower-level methods for transforming one image and extracting the final diffusion latent.

Parameters
----------

- ``model_id`` (*str*, optional): Hugging Face checkpoint. The default is ``"runwayml/stable-diffusion-v1-5"``.
- ``device`` (*str*, optional): execution device. If ``None``, CUDA is used when available and CPU otherwise.
- ``cache_dir`` (*str*, optional): directory for cached model files.
- ``dtype`` (*torch.dtype*, optional): model precision. The default is ``torch.float32``.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.diffusion import StableDiffusionImg2ImgExtractor

   extractor = StableDiffusionImg2ImgExtractor(
       model_id="runwayml/stable-diffusion-v1-5"
   )
   image, latent = extractor.transform_image(
       input_image="input.png",
       prompt="no change",
       strength=0,
       return_hidden_states=True,
   )

Methods
-------

preprocess_image
^^^^^^^^^^^^^^^^

``preprocess_image(image, max_size=512)`` accepts a PIL image or file path. It optionally resizes the longest side to ``max_size``, pads the image to multiples of eight, normalizes its pixels to ``[-1, 1]``, and returns a batched BCHW tensor on the configured device.

encode_prompt
^^^^^^^^^^^^^

``encode_prompt(prompt, negative_prompt=None)`` encodes one prompt or a list of prompts and returns the concatenated unconditional and conditional CLIP embeddings used for classifier-free guidance.

get_hidden_states
^^^^^^^^^^^^^^^^^

``get_hidden_states(input_image, prompt, strength=0.8, num_inference_steps=50, guidance_scale=7.5, negative_prompt=None, seed=None)`` runs the image-to-image denoising process and returns the final latent tensor. Despite the historical method name, the current implementation returns one latent tensor rather than a list of intermediate hidden states.

decode_latents
^^^^^^^^^^^^^^

``decode_latents(latents)`` decodes a latent tensor with the VAE and returns a normalized image tensor.

transform_image
^^^^^^^^^^^^^^^

``transform_image(input_image, prompt, strength=0.8, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5, seed=None, save_path=None, return_hidden_states=False)`` performs the complete transformation. It returns the image tensor by default, or ``(image, latent)`` when ``return_hidden_states=True``. If ``save_path`` is provided, it also saves the image.

save_image
^^^^^^^^^^

``save_image(image, save_path)`` converts the first image in a batched tensor to an 8-bit PIL image and writes it to disk.
