.. _ref_estimate_k_ipsi:

estimate_k_ipsi
===============

Description
-----------

The ``estimate_k_ipsi`` function estimates a longitudinal incremental-intervention curve with cross-fitting. It learns a Dynamic TarNet representation unless a precomputed representation is supplied, estimates the required propensity and backward-regression nuisance functions, and returns estimates, standard errors, confidence intervals, and influence-function values.

Data Arguments
--------------

- ``R`` (*np.ndarray*): per-segment vector representations. Supply ``[N, T, F]`` or ``[F, N, T]``. When ``R_video`` is provided, this is the aligned vector or text modality.
- ``W`` (*np.ndarray*): binary treatment history ``[N, T]``. Observed entries must be 0 or 1; trailing ``NaN`` entries indicate padding.
- ``Y`` (*np.ndarray*): one scalar outcome per unit, with shape ``[N]`` or ``[N, 1]``, or repeated outcomes with shape ``[N, T]`` exactly matching ``W``. Repeated outcomes can contain ``NaN`` where measurements are unavailable.
- ``delta_seq`` (*array-like*): positive treatment-odds multipliers. A scalar or ``[J]`` applies each multiplier at every segment; ``[J, T]`` supplies segment-specific paths.
- ``H`` (*np.ndarray*, optional): precomputed longitudinal representations with shape ``[N, T, d_z]`` or ``[d_z, N, T]``. Supplying ``H`` skips Dynamic TarNet representation learning.
- ``C`` (*np.ndarray*, optional): static covariates ``[N, P]`` or ``[N]``, or segment-varying covariates ``[N, T, P]``, ``[P, N, T]``, or ``[N, T]``.
- ``R_video`` (*np.ndarray*, optional): aligned video representations with shape ``[N, T, D, H, W]`` or ``[N, T, C_video, D, H, W]``. Supplying this argument enables multimodal mode.

Cross-Fitting Arguments
-----------------------

- ``K`` (*int*, optional): number of cross-fitting folds. The default is 5.
- ``sample_split_only`` (*bool*, optional): whether to estimate only one held-out fold. The default is ``False``. In this mode, non-held-out entries of scalar ``ifvals`` with shape ``[N, J]``, or repeated ``ifvals`` with shape ``[N, T, J]``, are ``NaN``.
- ``sample_split_fold`` (*int*, optional): one-indexed fold used when ``sample_split_only=True``. The default is 1. ``n_eff`` is an integer for a scalar outcome and has shape ``[T]`` for repeated outcomes.
- ``random_state`` (*int*, optional): seed for splitting, training, and bootstrap multipliers. The default is 42. Repeated-outcome estimation uses ``random_state + s`` for outcome segment ``s``.

Dynamic TarNet Arguments
------------------------

- ``architecture_y`` (*sequence of int*, optional): outcome-network widths ending in 1. The default is ``(16, 1)``.
- ``architecture_z`` (*sequence of int*, optional): representation-network widths. The default is ``(64, 32)``.
- ``nepoch`` (*int*, optional): maximum training epochs per fold. The default is 200.
- ``batch_size`` (*int*, optional): training batch size. The default is 32.
- ``lr`` (*float*, optional): learning rate. The default is ``2e-5``.
- ``dropout`` (*float*, optional): dropout probability. The default is 0.3.
- ``valid_perc`` (*float*, optional): validation fraction within each training fold. The default is 0.2.
- ``step_size`` (*int*, optional): reduce-on-plateau scheduler patience. ``None`` disables this scheduler.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``patience`` (*int*, optional): early-stopping patience. The default is 5.
- ``min_delta`` (*float*, optional): required validation-loss improvement. The default is 0.
- ``model_dir`` (*str*, optional): existing parent directory for fold-specific checkpoints. A nonexistent path raises ``ValueError``. Repeated-outcome estimation creates ``segment1``, ``segment2``, and subsequent subdirectories below it.
- ``device`` (*str*, *torch.device*, or *None*, optional): execution device. The default is ``"auto"``.
- ``verbose`` (*bool*, optional): whether to print progress. The default is ``True``.

Downstream Nuisance-Model Arguments
-----------------------------------

- ``eps_prob`` (*float* or *None*, optional): lower and upper probability clipping tolerance. The default is ``1e-6``.
- ``nn_hidden`` (*sequence of int*, optional): hidden widths of the propensity and regression MLPs. The default is ``(64, 32)``.
- ``nn_alpha`` (*float*, optional): downstream weight decay. The default is ``1e-4``.
- ``nn_lr`` (*float*, optional): downstream learning rate. The default is ``1e-3``.
- ``nn_lr_scheduler`` (*str*, optional): ``"none"`` or ``"adaptive"``. The aliases ``"plateau"`` and ``"reduce_on_plateau"`` are also accepted.
- ``nn_lr_scheduler_factor`` (*float*, optional): adaptive scheduler reduction factor. The default is 0.5.
- ``nn_lr_scheduler_patience`` (*int*, optional): adaptive scheduler patience. The default is 2.
- ``nn_lr_scheduler_min_lr`` (*float*, optional): minimum downstream learning rate. The default is ``1e-6``.
- ``nn_max_iter`` (*int*, optional): maximum downstream training epochs. The default is 300.
- ``nn_patience`` (*int*, optional): downstream early-stopping patience. The default is 5.
- ``nn_batch_size`` (*int* or *str*, optional): downstream batch size. The default is ``"auto"``.
- ``nn_dropout`` (*float*, optional): downstream dropout probability. The default is 0.

Multimodal Encoder Arguments
----------------------------

- ``text_input_dim`` (*int*, optional): vector feature width. It is inferred from ``R`` when ``None``.
- ``text_hidden_dims`` (*sequence of int*, optional): text-encoder hidden widths. The default is ``(1024, 256)``.
- ``text_out_dim`` (*int*, optional): encoded text width. The default is 128.
- ``video_in_channels`` (*int*, optional): video channel count for six-dimensional ``R_video`` and 1 for five-dimensional pooled input. The default is 1.
- ``video_channels`` (*sequence of int*, optional): 3D video-encoder widths. The default is ``(8, 16, 32)``.
- ``video_out_dim`` (*int*, optional): encoded video width. The default is 128.

Inference Argument
------------------

- ``n_boot`` (*int*, optional): number of Rademacher multiplier-bootstrap draws used for simultaneous 95% bands. The default 0 disables simultaneous bands. For repeated outcomes, the bands are simultaneous over outcome segments separately for each intervention.

Returns
-------

The function returns a dictionary with the following entries:

- ``delta``: intervention values supplied by the user.
- ``delta_paths``: intervention values expanded to ``[J, T]``.
- ``est``: estimated intervention curve.
- ``sigma``: estimated influence-function standard deviation.
- ``se``: standard error of every estimate.
- ``ll1`` and ``ul1``: pointwise 95% confidence limits.
- ``ll2`` and ``ul2``: simultaneous 95% confidence bands when ``n_boot > 0``; otherwise ``None``. Scalar-outcome bands are simultaneous across interventions. Repeated-outcome bands are simultaneous across outcome segments separately for each intervention.
- ``ifvals``: unit-level influence-function values.
- ``n_eff``: number of units used for inference.

For scalar ``Y`` and ``J`` interventions, ``est``, ``sigma``, ``se``, and the interval arrays have shape ``[J]``; ``ifvals`` has shape ``[N, J]``; and ``n_eff`` is an integer.

For repeated ``Y``, the estimator runs once for each outcome segment ``s``. It retains units with an observed treatment and finite outcome at ``s`` and uses histories only through ``s``. ``est``, ``sigma``, ``se``, and the interval arrays have shape ``[T, J]``; ``ifvals`` has shape ``[N, T, J]``; and ``n_eff`` has shape ``[T]``. Ineligible ``ifvals`` entries are ``NaN``. Every outcome segment must contain at least ``K`` eligible units.

A one-dimensional ``delta_seq`` defines constant interventions. For one time-varying schedule, supply ``[1, T]`` rather than ``[T]``. When ``delta_seq`` has shape ``[J, T]``, the estimate at outcome segment ``s`` uses the path prefix through ``s``.

Example Usage
-------------

.. code-block:: python

   import numpy as np
   from gpi_pack import estimate_k_ipsi

   result = estimate_k_ipsi(
       R=R_text,
       R_video=R_video,
       W=W,
       Y=Y,
       delta_seq=np.array([0.5, 1.0, 2.0]),
       K=5,
       architecture_y=[16, 1],
       architecture_z=[64, 32],
       n_boot=1000,
   )

   print(result["est"])
   print(result["ll1"], result["ul1"])

Repeated-outcome usage follows the same interface:

.. code-block:: python

   repeated_result = estimate_k_ipsi(
       R=R_text,
       R_video=R_video,
       W=W,
       Y=Y_repeated,  # [N, T], with NaN for unavailable outcomes
       delta_seq=np.array([0.5, 1.0, 2.0]),
       K=5,
       architecture_y=[16, 1],
       architecture_z=[64, 32],
       n_boot=1000,
   )

   print(repeated_result["est"].shape)     # [T, J]
   print(repeated_result["ifvals"].shape)  # [N, T, J]

.. note::
   A finite zero in ``W`` is an observed control value, not padding. Repeated outcomes are implemented as separate scalar-outcome fits at successive history prefixes, so the lower-level :ref:`ref_DynamicTarNet` and :ref:`ref_DynamicGPIHyperparameterTuner` interfaces remain scalar-outcome APIs.
