.. _ref_extract_images:

extract_images
==============

Description
-----------

The ``extract_images`` function processes one or more image-prompt pairs with Stable Diffusion. For every pair, it saves the final latent tensor and can also save the generated image. This is the high-level image-generation interface used by :ref:`generate_images`.

Arguments
---------

- ``images`` (*PIL.Image.Image*, *str*, or *list*): one image, one image path, or a list containing either type (**required**).
- ``prompts`` (*str* or *list of str*): one prompt or one prompt for each image (**required**).
- ``output_hidden_dir`` (*str*): directory for the latent ``.pt`` files (**required**).
- ``output_image_dir`` (*str*, optional): directory for generated PNG files. If ``None``, images are not saved.
- ``save_name`` (*str*, optional): generated-image filename prefix. The default is ``"gen"``.
- ``prefix_hidden`` (*str*, optional): latent filename prefix. The default is ``"hidden"``.
- ``model_id`` (*str*, optional): Stable Diffusion checkpoint. The default is ``"runwayml/stable-diffusion-v1-5"``.
- ``device`` (*str*, optional): execution device. The default selects CUDA when available.
- ``cache_dir`` (*str*, optional): directory for cached model files.
- ``strength`` (*float*, optional): image-to-image transformation strength from 0 to 1. The default is 0.
- ``num_inference_steps`` (*int*, optional): number of denoising steps. The default is 50.
- ``guidance_scale`` (*float*, optional): classifier-free guidance scale. The default is 7.5.
- ``negative_prompt`` (*str*, optional): content that the model should avoid.
- ``seed`` (*int*, optional): random seed for reproducibility.

Returns
-------

The function writes files and returns ``None``. Latents are saved as ``<prefix_hidden>_<index>.pt``. If requested, images are saved as ``<save_name>_<index>.png``.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.diffusion import extract_images

   extract_images(
       images=["input_0.png", "input_1.png"],
       prompts=["no change", "no change"],
       output_hidden_dir="outputs/hidden",
       output_image_dir="outputs/images",
       strength=0,
   )
