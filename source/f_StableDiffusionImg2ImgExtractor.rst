.. _ref_StableDiffusionImg2ImgExtractor:

StableDiffusionImg2ImgExtractor
===============================

Description
-----------

``StableDiffusionImg2ImgExtractor`` loads the VAE, one CLIP tokenizer and text
encoder, conditional UNet, and DDIM scheduler from a Stable Diffusion
checkpoint. It implements image-to-image denoising and returns the final
scaled diffusion latent. ``decode_latents`` unscales that tensor before
calling the VAE decoder.

.. code-block:: text

   StableDiffusionImg2ImgExtractor(
       model_id="runwayml/stable-diffusion-v1-5",
       device=None,
       cache_dir=None,
       dtype=torch.float32,
   )

The component layout and hard-coded latent scale ``0.18215`` are intended for
Stable Diffusion 1.x and 2.x checkpoints. The class does not implement Stable
Diffusion XL's second tokenizer/text encoder and added UNet conditioning, or
the Stable Diffusion 3/3.5 architectures.

Parameters
----------

- ``model_id`` (*str*, optional): compatible Hugging Face repository or local
  directory. The source default is ``"runwayml/stable-diffusion-v1-5"``, a
  now-deprecated repository whose access can fail. For Stable Diffusion 1.5,
  pass ``"stable-diffusion-v1-5/stable-diffusion-v1-5"`` explicitly.
- ``device`` (*str*, optional): execution device. ``None`` selects CUDA when
  available and CPU otherwise. Unlike Dynamic GPI, this class does not select
  Apple MPS automatically.
- ``cache_dir`` (*str*, optional): cache directory. The class creates it when
  supplied.
- ``dtype`` (*torch.dtype*, optional): dtype used for the VAE, text encoder,
  UNet, and input tensor. The default is ``torch.float32``.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.diffusion import StableDiffusionImg2ImgExtractor

   extractor = StableDiffusionImg2ImgExtractor(
       model_id="stable-diffusion-v1-5/stable-diffusion-v1-5"
   )
   image, latent = extractor.transform_image(
       input_image="input.png",
       prompt="no change",
       strength=0,
       seed=42,
       return_hidden_states=True,
   )

Methods
-------

preprocess_image
^^^^^^^^^^^^^^^^

``preprocess_image(image, max_size=512)`` accepts a PIL image or string file
path. When the longest side exceeds ``max_size``, it resizes while preserving
the aspect ratio. It then reflect-pads the spatial dimensions to multiples of
eight, converts to RGB, normalizes pixels to ``[-1, 1]``, and returns a
``[1, 3, H, W]`` tensor on the configured device and dtype. ``pathlib.Path``
is not recognized as a path by the current implementation.

encode_prompt
^^^^^^^^^^^^^

``encode_prompt(prompt, negative_prompt=None)`` accepts one prompt or a list
of prompts. It truncates each prompt to the CLIP tokenizer's maximum length
and returns the unconditional embeddings followed by the conditional
embeddings along the batch dimension. ``negative_prompt`` is one optional
string repeated for every prompt, not a list of per-prompt strings.

get_hidden_states
^^^^^^^^^^^^^^^^^

.. code-block:: text

   get_hidden_states(
       input_image,
       prompt,
       strength=0.8,
       num_inference_steps=50,
       guidance_scale=7.5,
       negative_prompt=None,
       seed=None,
   )

This method samples and scales the VAE latent. At positive ``strength``, it
adds noise at the corresponding DDIM timestep and denoises with classifier-free
guidance. It returns one final latent tensor. Despite the method name and its
old type annotation, it does not return intermediate hidden states or a tuple.

When ``seed`` is supplied, the method calls ``torch.manual_seed`` before VAE
sampling. At ``strength=0``, it returns the sampled initial latent after prompt
encoding and skips the scheduler and UNet. The intended strength range is 0 to
1, but the method does not explicitly validate it.

decode_latents
^^^^^^^^^^^^^^

``decode_latents(latents)`` divides by ``0.18215``, decodes with the VAE, and
returns the raw batched BCHW decoder tensor. It does not clamp or convert the
tensor to a PIL image.

transform_image
^^^^^^^^^^^^^^^

.. code-block:: text

   transform_image(
       input_image,
       prompt,
       strength=0.8,
       negative_prompt=None,
       num_inference_steps=50,
       guidance_scale=7.5,
       seed=None,
       save_path=None,
       return_hidden_states=False,
   )

This is the complete single-image workflow. It returns the decoded image
tensor by default, or ``(image, latent)`` when
``return_hidden_states=True``. If ``save_path`` is supplied, it also calls
``save_image``; the parent directory must already exist.

save_image
^^^^^^^^^^

``save_image(image, save_path)`` maps the first item of a batched decoder
tensor from the expected ``[-1, 1]`` range to 8-bit RGB and writes it with PIL.
Only the first batch item is saved.

.. warning::
   The class does not instantiate the safety checker used by the official
   Stable Diffusion pipeline.
