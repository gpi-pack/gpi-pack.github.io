.. _ref_extract_images:

extract_images
==============

Description
-----------

``extract_images`` is the high-level Stable Diffusion workflow. It processes
one or more image-prompt pairs, saves the final scaled diffusion latent, and
can save the decoded PNG image. The latent is divided by ``0.18215`` before it
is passed to the VAE decoder.

.. code-block:: text

   extract_images(
       images,
       prompts,
       output_hidden_dir,
       output_image_dir=None,
       save_name="gen",
       prefix_hidden="hidden",
       model_id="runwayml/stable-diffusion-v1-5",
       device=None,
       cache_dir=None,
       strength=0,
       num_inference_steps=50,
       guidance_scale=7.5,
       negative_prompt=None,
       seed=None,
   )

Arguments
---------

- ``images`` (*PIL.Image.Image*, *str*, or *list*): one image, one string file
  path, or a list containing either type (**required**). ``pathlib.Path`` is
  not accepted by the current normalization check.
- ``prompts`` (*str* or *list of str*): one prompt or a list of prompts
  (**required**). After a single string is converted to a one-element list,
  the prompt and image lists must have equal lengths; a prompt is not
  broadcast over multiple images.
- ``output_hidden_dir`` (*str*): latent output directory (**required**).
- ``output_image_dir`` (*str*, optional): decoded-image directory. ``None``
  skips PNG output.
- ``save_name`` (*str*, optional): image filename prefix. The default is
  ``"gen"``.
- ``prefix_hidden`` (*str*, optional): latent filename prefix. The default is
  ``"hidden"``.
- ``model_id`` (*str*, optional): checkpoint with the component layout
  expected by :ref:`ref_StableDiffusionImg2ImgExtractor`. The default is
  ``"runwayml/stable-diffusion-v1-5"``, but that repository is deprecated and
  access can fail. Pass
  ``"stable-diffusion-v1-5/stable-diffusion-v1-5"`` explicitly for the
  maintained Stable Diffusion 1.5 mirror.
- ``device`` (*str*, optional): execution device. ``None`` selects CUDA when
  available and CPU otherwise.
- ``cache_dir`` (*str*, optional): model cache directory.
- ``strength`` (*float*, optional): intended image-to-image strength from 0 to
  1. The default is 0.
- ``num_inference_steps`` (*int*, optional): DDIM schedule length. The default
  is 50.
- ``guidance_scale`` (*float*, optional): classifier-free guidance multiplier.
  The default is 7.5.
- ``negative_prompt`` (*str*, optional): one negative prompt applied to every
  pair.
- ``seed`` (*int*, optional): seed supplied to every image transformation.

Returns
-------

The function creates the requested output directories, writes files, and
returns ``None``. Although it builds an internal ``all_latents`` list, the
current implementation does not return that list.

Latents are written as ``<prefix_hidden>_<index>.pt`` and retain their PyTorch
device metadata. Images are written as ``<save_name>_<index>.png``. The
extractor is instantiated without a ``dtype`` argument, so this high-level
function always uses its float32 default.

At ``strength=0``, the saved value is the scaled VAE sample and neither the
UNet nor scheduler runs. It can still vary unless ``seed`` is set. Prompt,
guidance, negative-prompt, and inference-step values do not change that latent
at zero strength.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.diffusion import extract_images

   extract_images(
       images=["input_0.png", "input_1.png"],
       prompts=["no change", "no change"],
       output_hidden_dir="outputs/hidden",
       output_image_dir="outputs/images",
       model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
       strength=0,
       seed=42,
   )

.. warning::
   This custom workflow does not run the safety checker used by the official
   Stable Diffusion pipeline.
