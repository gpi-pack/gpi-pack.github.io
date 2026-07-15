.. _ref_Video3DEncoder:

Video3DEncoder
==============

Description
-----------

The ``Video3DEncoder`` class uses 3D convolutions and adaptive pooling to map one video latent volume to a fixed-width segment embedding. It is the video branch used by multimodal ``DynamicTarNetBase``. Every convolution uses a ``3 x 3 x 3`` kernel with padding and ReLU. Max pooling follows every convolution except the last, and adaptive average pooling reduces the remaining volume before the final projection.

Parameters
----------

- ``in_channels`` (*int*, optional): input channel count. The default is 1.
- ``channels`` (*sequence of int*, optional): convolutional output widths. The default is ``(8, 16, 32)``.
- ``out_dim`` (*int*, optional): final embedding width. The default is 128.
- ``dropout`` (*float*, optional): output dropout probability. The default is 0.

Returns
-------

``forward(x)`` accepts ``[B, C, D, H, W]`` and returns a nonnegative tensor with shape ``[B, out_dim]`` after the projection, ReLU, and optional dropout. ``C`` must equal ``in_channels``. All channel and output widths must be positive, and ``channels`` must contain at least one width. Because there are ``len(channels) - 1`` max-pooling operations, each of ``D``, ``H``, and ``W`` should be at least ``2 ** (len(channels) - 1)``.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.dyn_gpi import Video3DEncoder

   encoder = Video3DEncoder(in_channels=1, out_dim=128)
   encoded = encoder(torch.randn(8, 1, 16, 40, 60))
   print(encoded.shape)  # [8, 128]
