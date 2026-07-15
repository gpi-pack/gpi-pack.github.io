.. _ref_TarNetBase:

TarNetBase
==========

Description
-----------

The ``TarNetBase`` class implements the neural architecture used for static treatment-effect estimation. It learns a shared representation from the input, optionally appends observed confounders, concatenates the requested treatment value, and predicts the corresponding outcome with one treatment-conditioned outcome network. An optional convolutional front end supports image-shaped inputs.

Parameters
----------

- ``sizes_z`` (*sequence of int*, optional): layer widths of the shared representation network. The default is ``[2048]``.
- ``sizes_y`` (*sequence of int*, optional): additional layer widths of the outcome network. The final width is the outcome dimension. The default is ``[200, 1]``. The implementation prepends an outcome layer of width ``sizes_z[-1]``.
- ``dropout`` (*float*, optional): dropout probability. The default is ``None``.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``conv_layers`` (*list of dict*, optional): convolutional layer specifications applied before the shared representation. The first entry must include ``in_channels`` and every entry must include ``out_channels``. The implementation recognizes exactly ``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, and ``bias`` as additional ``Conv2d`` options; unsupported keys such as ``padding_mode`` are ignored. A specification can also contain ``spectral_norm`` and a ``pool`` dictionary. Pooling defaults to max pooling; set ``pool["type"]`` to ``"avg"`` for average pooling.
- ``conv_activation`` (*callable*, optional): activation factory for convolutional blocks. The default is ``torch.nn.ReLU``; use ``None`` to omit these activations.

forward
-------

``forward(inputs, treatments, confounders=None)`` returns ``(y, fr)``. Without a convolutional front end, ``inputs`` has shape ``[B, F]``; with one, it has shape ``[B, C, H, W]``. ``treatments`` must have shape ``[B, 1]``. ``y`` has shape ``[B, sizes_y[-1]]`` and contains the outcome prediction under each supplied treatment value. Without additional confounders, ``fr`` has shape ``[B, sizes_z[-1]]``. Optional ``confounders`` must have shape ``[B, P]`` and are appended to ``fr``, giving it shape ``[B, sizes_z[-1] + P]``.

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
