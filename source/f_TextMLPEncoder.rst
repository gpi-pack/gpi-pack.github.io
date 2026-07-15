.. _ref_TextMLPEncoder:

TextMLPEncoder
==============

Description
-----------

The ``TextMLPEncoder`` class maps each vector or text representation to a fixed-width embedding with a multilayer perceptron. It is the text branch used by multimodal ``DynamicTarNetBase``. Each hidden layer is followed by ReLU and optional dropout; the final projection to ``out_dim`` is linear.

Parameters
----------

- ``input_dim`` (*int*): width of the input representation.
- ``hidden_dims`` (*sequence of int*, optional): hidden-layer widths. The default is ``(1024, 256)``.
- ``out_dim`` (*int*, optional): output width. The default is 128.
- ``dropout`` (*float*, optional): dropout probability after hidden layers. The default is 0.

Returns
-------

``forward(x)`` maps the tensor's last dimension from ``input_dim`` to ``out_dim`` while preserving all leading dimensions. For example, both ``[B, input_dim]`` and ``[B, T, input_dim]`` inputs are accepted by the underlying linear layers. All input, hidden, and output widths must be positive; ``hidden_dims=()`` gives a single direct linear projection.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.dyn_gpi import TextMLPEncoder

   encoder = TextMLPEncoder(input_dim=4096, out_dim=128)
   encoded = encoder(torch.randn(32, 4096))
   print(encoded.shape)  # [32, 128]
