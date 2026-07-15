.. _ref_TextMLPEncoder:

TextMLPEncoder
==============

Description
-----------

The ``TextMLPEncoder`` class maps each vector or text representation to a fixed-width embedding with a multilayer perceptron. It is the text branch used by multimodal ``DynamicTarNetBase``.

Parameters
----------

- ``input_dim`` (*int*): width of the input representation.
- ``hidden_dims`` (*sequence of int*, optional): hidden-layer widths. The default is ``(1024, 256)``.
- ``out_dim`` (*int*, optional): output width. The default is 128.
- ``dropout`` (*float*, optional): dropout probability after hidden layers. The default is 0.

Returns
-------

``forward(x)`` maps the tensor's last dimension from ``input_dim`` to ``out_dim`` while preserving the leading dimensions.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.dyn_gpi import TextMLPEncoder

   encoder = TextMLPEncoder(input_dim=4096, out_dim=128)
   encoded = encoder(torch.randn(32, 4096))
   print(encoded.shape)  # [32, 128]
