.. _ref_DynamicTarNetBase:

DynamicTarNetBase
=================

Description
-----------

The ``DynamicTarNetBase`` class implements the neural representation and outcome networks used by Dynamic GPI. It masks padded segment positions, learns one representation per observed segment, combines the representation history with the treatment history and optional covariates, and predicts one outcome vector for each unit. :ref:`ref_DynamicTarNet` requires ``outcome_dim=1`` for each fit. :ref:`ref_estimate_k_ipsi` supports repeated outcomes by fitting a separate scalar model for each outcome segment rather than one joint multi-output head.

Parameters
----------

- ``sizes_z`` (*sequence of int*): widths of the learned representation network.
- ``sizes_y`` (*sequence of int*): widths of the outcome network. The final width must equal ``outcome_dim``.
- ``outcome_dim`` (*int*, optional): outcome width. The supported high-level workflow uses 1.
- ``dropout`` (*float*, optional): dropout probability. The default is ``None``.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``include_Si_in_head`` (*bool*, optional): whether to include the observed sequence length in the outcome network. The default is ``True``.
- ``include_Si_in_rep`` (*bool*, optional): whether to include sequence length in the representation network. The default is ``False``.
- ``include_C_in_head`` (*bool*, optional): whether to include covariates in the outcome network. The default is ``True``.
- ``include_C_in_rep`` (*bool*, optional): whether to include covariates in the representation network. The default is ``False``.
- ``multimodal`` (*bool*, optional): whether to enable the text and video encoders. The default is ``False``.
- ``text_input_dim`` (*int*, optional): input width required in multimodal mode.
- ``text_hidden_dims`` (*sequence of int*, optional): text-encoder hidden widths. The default is ``(1024, 256)``.
- ``text_out_dim`` (*int*, optional): encoded text width. The default is 128.
- ``video_in_channels`` (*int*, optional): input channels for the 3D video encoder. Keep the default 1 for pooled Cosmos input shaped ``[N, T, D, H, W]``, where the Cosmos feature width is treated as depth ``D``. For unpooled six-dimensional input ``[N, T, C_video, D, H, W]``, set this value to ``C_video``.
- ``video_channels`` (*sequence of int*, optional): video-encoder convolutional widths. The default is ``(8, 16, 32)``.
- ``video_out_dim`` (*int*, optional): encoded video width. The default is 128.

forward
-------

``forward(r, w, mask, c=None, return_rep=False, r_video=None)`` returns ``y_pred`` with shape ``[B, outcome_dim]``. With ``return_rep=True``, it returns ``(y_pred, h, h_flat)``, where ``h`` has shape ``[B, T, sizes_z[-1]]`` and ``h_flat`` has shape ``[B, T * sizes_z[-1]]``.

In vector-only mode, the model-level input ``r`` has shape ``[F, B, T]``. In multimodal mode, ``r`` has shape ``[B, T, F]`` and ``r_video`` has shape ``[B, T, D, H, W]`` or ``[B, T, C, D, H, W]``.

Example Usage
-------------

Most users should use :ref:`ref_DynamicTarNet`, which prepares these tensors and trains the base model.

.. code-block:: python

   from gpi_pack.dyn_gpi import DynamicTarNetBase

   model = DynamicTarNetBase(
       sizes_z=[64, 32],
       sizes_y=[16, 1],
   )
