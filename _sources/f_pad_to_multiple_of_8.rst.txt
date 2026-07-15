.. _ref_pad_to_multiple_of_8:

pad_to_multiple_of_8
====================

Description
-----------

The ``pad_to_multiple_of_8`` function symmetrically pads a PIL image until its height and width are both divisible by eight. Stable Diffusion's VAE requires compatible spatial dimensions, so the image extractor calls this function during preprocessing.

Arguments
---------

- ``img`` (*PIL.Image.Image*): input image.
- ``pad_mode`` (*str*, optional): NumPy padding mode. The default is ``"reflect"``.

Returns
-------

- *PIL.Image.Image*: the original image when no padding is needed, or an RGB image with the required padding.

When padding is applied, the function prints the original dimensions and the
new padded dimensions. It does not print anything when the input dimensions
are already divisible by eight.

Example Usage
-------------

.. code-block:: python

   from PIL import Image
   from gpi_pack.diffusion import pad_to_multiple_of_8

   image = Image.open("input.png")
   padded = pad_to_multiple_of_8(image)
   print(padded.size)
