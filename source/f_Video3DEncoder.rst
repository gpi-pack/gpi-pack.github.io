.. _ref_Video3DEncoder:

Video3DEncoder
==============

Description
-----------

The ``Video3DEncoder`` class uses 3D convolutions and adaptive pooling to map one video latent volume to a fixed-width segment embedding. It is the video branch used by multimodal ``DynamicTarNetBase``.

Parameters
----------

- ``in_channels`` (*int*, optional): input channel count. The default is 1.
- ``channels`` (*sequence of int*, optional): convolutional output widths. The default is ``(8, 16, 32)``.
- ``out_dim`` (*int*, optional): final embedding width. The default is 128.
- ``dropout`` (*float*, optional): output dropout probability. The default is 0.

Returns
-------

``forward(x)`` accepts ``[B, C, D, H, W]`` and returns a tensor with shape ``[B, out_dim]``.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.dyn_gpi import Video3DEncoder

   encoder = Video3DEncoder(in_channels=1, out_dim=128)
   encoded = encoder(torch.randn(8, 1, 16, 40, 60))
   print(encoded.shape)  # [8, 128]
