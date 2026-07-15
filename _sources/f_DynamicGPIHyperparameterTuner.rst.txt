.. _ref_DynamicGPIHyperparameterTuner:

DynamicGPIHyperparameterTuner
=============================

Description
-----------

The ``DynamicGPIHyperparameterTuner`` class uses Optuna to select hyperparameters for the Dynamic TarNet outcome model. It minimizes the best held-out factual-outcome MSE reached during each trial and can refit the best configuration. ``DynamicTarNetHyperparameterTuner`` is an alias of the same class.

Parameters
----------

- ``R`` (*np.ndarray*): vector or text representations shaped ``[N, T, F]`` or ``[F, N, T]``.
- ``W`` (*np.ndarray*): binary treatment history ``[N, T]``. Trailing ``NaN`` values indicate padding.
- ``Y`` (*np.ndarray*): one scalar outcome per unit, shaped ``[N]`` or ``[N, 1]``.
- ``C`` (*np.ndarray*, optional): static covariates ``[N]`` or ``[N, P]``, or segment-varying covariates ``[N, T]``, ``[N, T, P]``, or ``[P, N, T]``.
- ``R_video`` (*np.ndarray*, optional): aligned video representations shaped ``[N, T, D, H, W]`` or ``[N, T, C_video, D, H, W]``. Supplying it enables multimodal tuning.
- ``nepoch`` (*value or sequence*, optional): epoch candidates. The default is ``(100, 200)``.
- ``batch_size`` (*value or sequence*, optional): batch-size candidates. The default is ``(16, 32)``.
- ``lr`` (*value or sequence*, optional): fixed value or log-scale continuous lower and upper search bounds. The default bounds are ``(1e-5, 1e-3)``.
- ``dropout`` (*value or sequence*, optional): fixed value or linear-scale continuous lower and upper search bounds. The default bounds are ``(0.0, 0.3)``.
- ``architecture_y`` (*sequence of architectures*, optional): outcome-network candidates. The default is ``((16, 1), (32, 16, 1))``.
- ``architecture_z`` (*sequence of architectures*, optional): representation-network candidates. The default is ``((64, 32), (128, 64))``.
- ``text_hidden_dims`` (*sequence of architectures*, optional): text-encoder candidates. The default is ``((1024, 256),)``.
- ``text_out_dim`` (*value or sequence*, optional): encoded-text-width candidates. The default is ``(128,)``.
- ``video_channels`` (*sequence of architectures*, optional): video-encoder candidates. The default is ``((8, 16, 32),)``.
- ``video_out_dim`` (*value or sequence*, optional): encoded-video-width candidates. The default is ``(128,)``.
- ``valid_perc`` (*float*, optional): validation fraction. The default is 0.2.
- ``random_state`` (*int*, optional): seed. The default is 42.
- ``device`` (*str*, *torch.device*, or *None*, optional): execution device. The default is ``"auto"``.
- ``model_dir`` (*str*, optional): checkpoint directory used only by ``fit_best``. If it is nonempty, the directory must exist when refitting.
- ``verbose`` (*bool*, optional): whether to print model progress. The default is ``False``.

Every unit must have at least one observed treatment, and padding in ``W`` must be trailing. Observed treatments must be finite zero or one. Observed ``R``, time-varying ``C``, and ``R_video`` values must be finite; padded positions are replaced with zero. A two-dimensional ``C`` whose shape is exactly ``[N, T]`` is interpreted as a scalar time-varying covariate.

Search-Space Behavior
---------------------

``nepoch``, ``batch_size``, ``text_out_dim``, and ``video_out_dim`` are categorical choices. ``lr`` is sampled log-uniformly between the smallest and largest supplied values, while ``dropout`` is sampled uniformly between its smallest and largest values. Additional interior values in either continuous sequence do not create categorical choices.

A flat sequence such as ``architecture_y=(16, 1)`` is one fixed architecture. Use a nested sequence such as ``architecture_y=((16, 1), (32, 16, 1))`` to define multiple categorical candidates. Architecture strings such as ``"[16, 1]"`` remain accepted, but native sequences are recommended. Every architecture width must be positive, and each ``architecture_y`` candidate must end in 1.

The text- and video-encoder choices are sampled only when ``R_video`` enables multimodal mode. In this mode, the tuner infers ``text_input_dim`` and ``video_in_channels`` from the supplied arrays and includes both values in ``best_params_``. With ``len(video_channels) - 1`` pooling operations, every video volume dimension must be at least ``2 ** (len(video_channels) - 1)``.

Example Usage
-------------

.. code-block:: python

   from gpi_pack import DynamicGPIHyperparameterTuner

   tuner = DynamicGPIHyperparameterTuner(
       R=R_text,
       R_video=R_video,
       W=W,
       Y=Y,
       architecture_y=((16, 1), (32, 16, 1)),
       architecture_z=((64, 32), (128, 64)),
   )

   study = tuner.tune(n_trials=50, refit=True)
   print(tuner.best_params_)
   best_model = tuner.best_model_

Methods
-------

- ``objective(trial)`` samples a configuration, trains it on a fixed split, and returns its best validation MSE.
- ``tune(n_trials=50, timeout=None, study=None, sampler=None, pruner=None, n_jobs=1, refit=False, **optimize_kwargs)`` creates or uses an Optuna study, performs minimization, optionally refits, and returns the study.
- ``fit_best(study=None)`` fits and returns a :ref:`ref_DynamicTarNet` using the best resolved parameters.

When creating a study, ``tune`` uses a seeded TPE sampler and a median pruner with five startup trials. A supplied study must be single-objective with direction ``"minimize"``. The default ``n_jobs=1`` avoids thread-shared PyTorch random-number and accelerator state; other nonzero values emit a reproducibility warning.

After tuning, ``study_``, ``best_trial_``, ``best_score_``, and ``best_params_`` describe the selected trial. ``best_model_`` remains ``None`` unless ``refit=True`` or ``fit_best`` is called. ``fit_best`` creates a fresh model with the selected settings and repeats the same seeded training/validation split; it does not retrain on all observations. Use ``tuner.best_params_``, rather than ``study.best_params``, because the resolved dictionary includes fixed choices and converts categorical architecture labels back to integer lists. It can be passed directly to :ref:`ref_estimate_k_ipsi`.

The tuner fixes ``outcome_dim=1``, disables batch normalization, and uses early-stopping patience 5 with ``min_delta=0``. It optimizes the factual-outcome Dynamic TarNet only, not the propensity and backward-regression models used later by :ref:`ref_estimate_k_ipsi`. For strict causal cross-fitting, tune on an independent sample or inside each outer training fold to avoid leaking held-out outcomes.

.. note::
   This tuner accepts one scalar outcome per unit and does not accept repeated ``Y`` with shape ``[N, T]``. A repeated-outcome call to :ref:`ref_estimate_k_ipsi` uses one shared hyperparameter configuration for all outcome segments. You can therefore tune a prespecified scalar endpoint or representative eligible history prefix and reuse that configuration. If you require a separately tuned configuration at every segment, run the scalar estimator separately for those prefixes.

Install version 0.2.1 and its optional Optuna dependency from PyPI:

.. code-block:: bash

   python -m pip install --upgrade "gpi-pack[tune]"
