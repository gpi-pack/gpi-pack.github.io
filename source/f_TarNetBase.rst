.. _ref_TarNetBase:

TarNetBase
===========

Description
-----------

The ``TarNetBase`` class implements the neural architecture used for static treatment-effect estimation. It learns a shared representation from the input, optionally appends observed confounders, concatenates the requested treatment value, and predicts the corresponding outcome with one treatment-conditioned outcome network. An optional convolutional front end supports image-shaped inputs.

Parameters
----------

- ``sizes_z`` (*sequence of int*, optional): layer widths of the shared representation network. The default is ``[2048]``.
- ``sizes_y`` (*sequence of int*, optional): layer widths of the outcome network. The default is ``[200, 1]``.
- ``dropout`` (*float*, optional): dropout probability. The default is ``None``.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``conv_layers`` (*list of dict*, optional): convolutional layer specifications applied before the shared representation. The first entry must include ``in_channels``; each entry includes ``out_channels`` and can include standard ``Conv2d`` options, ``spectral_norm``, and a pooling specification.
- ``conv_activation`` (*callable*, optional): activation factory for convolutional blocks. The default is ``torch.nn.ReLU``.

forward
-------

``forward(inputs, treatments, confounders=None)`` returns ``(y, fr)``. ``y`` has shape ``[B, sizes_y[-1]]`` and contains the outcome prediction under each supplied treatment value. Without additional confounders, ``fr`` has shape ``[B, sizes_z[-1]]``; supplied confounders are appended to its last dimension.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.TNutil import TarNetBase

   model = TarNetBase(
       sizes_z=[2048],
       sizes_y=[200, 1],
       dropout=0.2,
   )

   y_pred, representation = model(
       inputs=torch.randn(16, 4096),
       treatments=torch.zeros(16, 1),
   )
