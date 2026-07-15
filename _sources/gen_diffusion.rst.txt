.. _generate_images:

Generating Images with Stable Diffusion
=======================================

For image GPI, **gpi_pack** transforms or reconstructs an input image and saves
the final scaled latent in the diffusion process. The current
implementation uses the component layout of Stable Diffusion 1.x and 2.x: one
VAE, one CLIP tokenizer and text encoder, a conditional UNet, and a DDIM
scheduler.

How the Image Representation Is Constructed
--------------------------------------------

The extractor performs the following operations:

1. It converts the input to RGB, limits the longest side to 512 pixels while
   preserving the aspect ratio, pads both dimensions to multiples of eight,
   and normalizes pixels to ``[-1, 1]``.

2. The VAE encoder produces a latent distribution. The implementation samples
   from this distribution and multiplies the sample by ``0.18215``.

3. When ``strength`` is greater than zero, the extractor adds noise and runs
   DDIM denoising with classifier-free text guidance.

4. It saves the final latent in the scaled diffusion space. Before decoding,
   ``decode_latents`` divides this tensor by ``0.18215`` and then calls the
   VAE. For a 512 by 512 Stable Diffusion 1.5 input, the typical batched shape
   is ``[1, 4, 64, 64]``. Other padded input sizes produce corresponding
   spatial dimensions.

The method name ``get_hidden_states`` is historical: it returns one final
latent tensor, not a collection of UNet hidden states.

Using ``extract_images``
------------------------

``extract_images`` is the high-level interface. It processes one image-prompt
pair at a time, saves every final latent, and optionally saves the decoded
image.

.. code-block:: python

   from PIL import Image
   from gpi_pack.diffusion import extract_images

   image = Image.open("input.png").convert("RGB")

   extract_images(
       images=image,
       prompts="no change",
       output_hidden_dir="outputs/image_hidden",
       output_image_dir="outputs/reconstructed",
       save_name="image",
       prefix_hidden="hidden",
       model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
       strength=0,
       seed=42,
   )

This writes ``outputs/image_hidden/hidden_0.pt`` and
``outputs/reconstructed/image_0.png``. The function creates both output
directories and returns ``None``.

At ``strength=0``, the function skips noise addition and the UNet denoising
loop. The prompt, negative prompt, guidance scale, and number of inference
steps therefore do not affect the returned latent. The decoded image is still
a VAE reconstruction rather than a pixel-identical copy, and the latent is
still sampled from the VAE distribution. Set ``seed`` when you need to repeat
that sample.

Transforming Multiple Images
----------------------------

For multiple inputs, supply exactly one prompt per image. A single prompt is
not broadcast over a list of images:

.. code-block:: python

   extract_images(
       images=["input_0.png", "input_1.png"],
       prompts=["render as a watercolor", "render as a pencil drawing"],
       output_hidden_dir="outputs/transformed_hidden",
       output_image_dir="outputs/transformed_images",
       strength=0.6,
       num_inference_steps=50,
       guidance_scale=7.5,
       negative_prompt="blurry",
       seed=42,
   )

With positive ``strength``, a larger value uses more of the denoising schedule
and generally permits a larger departure from the input. The public function
documents the intended range as 0 to 1, but the implementation does not
explicitly validate that range; keep the value within it.

Arguments
---------

- ``images``: one PIL image, one string file path, or a list containing those
  types (**required**). ``pathlib.Path`` is not accepted by the current type
  check.
- ``prompts``: one string or a list with exactly the same length as ``images``
  after normalization (**required**).
- ``output_hidden_dir``: directory for saved latent tensors (**required**).
- ``output_image_dir``: optional directory for decoded PNG images. ``None``
  skips image files.
- ``save_name``: image filename prefix. The default is ``"gen"``.
- ``prefix_hidden``: latent filename prefix. The default is ``"hidden"``;
  the function adds ``_<index>.pt`` itself.
- ``model_id``: compatible Stable Diffusion checkpoint. The source default is
  ``"runwayml/stable-diffusion-v1-5"``, but that repository is now
  deprecated and access can fail. Pass the maintained mirror
  ``"stable-diffusion-v1-5/stable-diffusion-v1-5"`` explicitly, as in the
  example.
- ``device``: PyTorch device. ``None`` selects CUDA when available and CPU
  otherwise. The high-level function uses float32 even on CUDA.
- ``cache_dir``: optional Hugging Face cache directory.
- ``strength``: intended transformation strength from 0 to 1. The default is
  0.
- ``num_inference_steps``: DDIM schedule length. The default is 50.
- ``guidance_scale``: classifier-free guidance multiplier. The default is 7.5.
- ``negative_prompt``: optional single negative-prompt string.
- ``seed``: optional seed passed to ``torch.manual_seed`` before each image.

The saved tensor retains the extractor device metadata. Load it portably with
``torch.load(path, map_location="cpu", weights_only=True)`` or use
:ref:`ref_load_hiddens`, which maps tensors to its requested device and returns
float32 CPU data.

For the lower-level class and exact method behavior, see
:ref:`ref_StableDiffusionImg2ImgExtractor`. For the complete high-level
function reference, see :ref:`ref_extract_images`.
