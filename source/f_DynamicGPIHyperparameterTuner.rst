.. _ref_DynamicGPIHyperparameterTuner:

DynamicGPIHyperparameterTuner
=============================

Description
-----------

The ``DynamicGPIHyperparameterTuner`` class uses Optuna to select hyperparameters for the Dynamic TarNet outcome model. It minimizes the best held-out factual-outcome MSE reached during each trial and can refit the best configuration. ``DynamicTarNetHyperparameterTuner`` is an alias of the same class.

Parameters
----------

- ``R`` (*np.ndarray*): vector or text representations.
- ``W`` (*np.ndarray*): binary treatment history with trailing ``NaN`` padding.
- ``Y`` (*np.ndarray*): one scalar outcome per unit.
- ``C`` (*np.ndarray*, optional): static or segment-varying covariates.
- ``R_video`` (*np.ndarray*, optional): aligned video representations; supplying it enables multimodal tuning.
- ``nepoch`` (*value or sequence*, optional): epoch candidates. The default is ``(100, 200)``.
- ``batch_size`` (*value or sequence*, optional): batch-size candidates. The default is ``(16, 32)``.
- ``lr`` (*value or sequence*, optional): fixed value or continuous lower and upper search bounds. The default bounds are ``(1e-5, 1e-3)``.
- ``dropout`` (*value or sequence*, optional): fixed value or continuous lower and upper search bounds. The default bounds are ``(0.0, 0.3)``.
- ``architecture_y`` (*sequence of architectures*, optional): outcome-network candidates. The default is ``((16, 1), (32, 16, 1))``.
- ``architecture_z`` (*sequence of architectures*, optional): representation-network candidates. The default is ``((64, 32), (128, 64))``.
- ``text_hidden_dims`` (*sequence of architectures*, optional): text-encoder candidates. The default is ``((1024, 256),)``.
- ``text_out_dim`` (*value or sequence*, optional): encoded-text-width candidates. The default is ``(128,)``.
- ``video_channels`` (*sequence of architectures*, optional): video-encoder candidates. The default is ``((8, 16, 32),)``.
- ``video_out_dim`` (*value or sequence*, optional): encoded-video-width candidates. The default is ``(128,)``.
- ``valid_perc`` (*float*, optional): validation fraction. The default is 0.2.
- ``random_state`` (*int*, optional): seed. The default is 42.
- ``device`` (*str*, *torch.device*, or *None*, optional): execution device. The default is ``"auto"``.
- ``model_dir`` (*str*, optional): existing checkpoint directory used when refitting. A nonexistent path raises ``ValueError``.
- ``verbose`` (*bool*, optional): whether to print model progress. The default is ``False``.

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

After tuning, ``best_params_`` can be passed directly to :ref:`ref_estimate_k_ipsi`. For strict causal cross-fitting, tune on an independent sample or inside each outer training fold to avoid leaking held-out outcomes.

.. note::
   This tuner accepts one scalar outcome per unit and does not accept repeated ``Y`` with shape ``[N, T]``. For repeated-outcome estimation, tune one eligible outcome-history prefix at a time or reuse a scalar-tuned parameter configuration with :ref:`ref_estimate_k_ipsi`.

Install the optional dependency with ``pip install "gpi_pack[tune]"``.
