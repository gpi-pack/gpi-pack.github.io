.. _ref_DynamicTarNet:

DynamicTarNet
=============

Description
-----------

The ``DynamicTarNet`` class trains and evaluates :ref:`ref_DynamicTarNetBase` for padded sequential data. It supports vector-only inputs and a multimodal combination of aligned vector or text representations with video latent volumes.

Parameters
----------

- ``architecture_y`` (*sequence of int*, optional): outcome-network widths. The default is ``(16, 1)``.
- ``architecture_z`` (*sequence of int*, optional): representation-network widths. The default is ``(64, 32)``.
- ``outcome_dim`` (*int*, optional): scalar outcome width. The supported value is 1.
- ``epochs`` (*int*, optional): maximum training epochs. The default is 200.
- ``batch_size`` (*int*, optional): training batch size. The default is 32.
- ``learning_rate`` (*float*, optional): AdamW learning rate. The default is ``2e-5``.
- ``dropout`` (*float*, optional): dropout probability in the representation and outcome networks and, in multimodal mode, both modality encoders. The default is ``None``.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``step_size`` (*int*, optional): reduce-on-plateau scheduler patience. ``None`` disables the scheduler.
- ``patience`` (*int*, optional): early-stopping patience. The default is 5.
- ``min_delta`` (*float*, optional): required validation-loss improvement. The default is 0.01.
- ``model_dir`` (*str*, optional): existing directory for ``best_DynamicTarNet.pth``. A nonexistent path raises ``ValueError``.
- ``verbose`` (*bool*, optional): whether to print progress. The default is ``True``.
- ``random_state`` (*int*, optional): training and split seed. The default is 42.
- ``include_Si_in_head``, ``include_Si_in_rep``, ``include_C_in_head``, ``include_C_in_rep`` (*bool*, optional): control whether sequence length and covariates enter the representation and outcome networks. Their defaults are ``True``, ``False``, ``True``, and ``False``, respectively.
- ``device`` (*str*, *torch.device*, or *None*, optional): execution device. The default ``"auto"`` selects CUDA, MPS, or CPU when available.
- ``multimodal`` (*bool*, optional): whether to enable aligned text/vector and video encoders. The default is ``False``.
- ``text_input_dim`` (*int*, optional): vector/text input width in multimodal mode. The default is 4096 and must match the last dimension of ``R``.
- ``text_hidden_dims`` (*sequence of int*, optional): text-encoder hidden widths. The default is ``(1024, 256)``.
- ``text_out_dim`` (*int*, optional): encoded text width. The default is 128.
- ``video_in_channels`` (*int*, optional): video input channels. The default is 1. This must match ``C_video`` for input shaped ``[N, T, C_video, D, H, W]``; five-dimensional input receives a singleton channel automatically.
- ``video_channels`` (*sequence of int*, optional): 3D encoder widths. The default is ``(8, 16, 32)``.
- ``video_out_dim`` (*int*, optional): encoded video width. The default is 128.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.dyn_gpi import DynamicTarNet

   model = DynamicTarNet(
       architecture_y=[16, 1],
       architecture_z=[64, 32],
       epochs=200,
       device="auto",
   )

   best_loss = model.fit(R, Y, W, mask, valid_perc=0.2)
   predictions, h, h_flat = model.predict(
       R, W, mask, return_rep=True
   )

Methods
-------

- ``reset_model()`` rebuilds the neural model and optimizer using the stored configuration.
- ``create_dataloaders(r_train, r_test, w_train, w_test, y_train, y_test, mask_train, mask_test, c_train=None, c_test=None, r_video_train=None, r_video_test=None)`` validates the vector, treatment, mask, outcome, optional covariate, and optional video arrays and constructs the training and validation loaders. The video arguments are keyword-only.
- ``fit(R, Y, W, mask, C=None, valid_perc=0.2, plot_loss=True, *, R_video=None, epoch_callback=None)`` makes a seeded unit-level training/validation split, trains with early stopping, restores the parameters with the smallest validation MSE, and returns that loss as a float. When provided, ``epoch_callback(epoch, valid_loss)`` is called after every epoch.
- ``validate_step()`` evaluates the current validation loader and returns the sample-weighted MSE as a scalar CPU tensor.
- ``predict(R, W, mask, C=None, return_rep=False, *, R_video=None)`` returns CPU outcome tensors with shape ``[N, 1]``. With ``return_rep=True``, it returns ``(predictions, h, h_flat)``, where ``h`` is ``[N, T, architecture_z[-1]]`` and ``h_flat`` is ``[N, T * architecture_z[-1]]``. Masked entries of ``h`` are zero.

The treatment ``W`` and mask must both have shape ``[N, T]``. Masks must contain zero or one and be left aligned: an observed position cannot follow padding. ``W`` must be finite and binary wherever the mask is one; masked values are replaced with zero. ``R`` accepts ``[N, T, F]`` or ``[F, N, T]``. In multimodal mode, ``R_video`` is required and accepts ``[N, T, D, H, W]`` or ``[N, T, C, D, H, W]``. In vector-only mode, passing ``R_video`` raises ``ValueError``.

Optional covariates can be static ``[N]`` or ``[N, P]``, or time-varying ``[N, T]``, ``[N, T, P]``, or ``[P, N, T]``. A two-dimensional array whose shape is exactly ``[N, T]`` is interpreted as a scalar time-varying covariate.

.. note::
   ``DynamicTarNet`` accepts one scalar outcome per unit, with ``Y`` shaped ``[N]`` or ``[N, 1]``. Pass repeated outcomes shaped ``[N, T]`` to :ref:`ref_estimate_k_ipsi`, which fits the scalar model separately for successive history prefixes.
