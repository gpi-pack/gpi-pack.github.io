.. _ref_TarNetHyperparameterTuner:

TarNetHyperparameterTuner
=========================

Description
-----------

The ``TarNetHyperparameterTuner`` class uses Optuna to tune the static TarNet outcome model. It minimizes the best held-out factual-outcome MSE reached by each trial and can refit the best configuration.

Parameters
----------

- ``T`` and ``Y`` (*array-like*): one treatment and scalar outcome per observation (**required**).
- ``R`` (*array-like*): representations with leading observation dimension (**required**). These are usually ``[N, F]``, or ``[N, C, H, W]`` when a convolutional front end is configured.
- ``C`` (*array-like*, optional): observed confounders. A one-dimensional input is reshaped to ``[N, 1]``.
- ``formula_C`` (*str*, optional): Patsy formula used with ``data`` to construct confounders. Supply either ``C`` or ``formula_C``.
- ``data`` (*pandas.DataFrame*, optional): data used by ``formula_C``.
- ``epoch`` (*value or sequence*, optional): epoch candidates. The default is ``(100, 200)``.
- ``batch_size`` (*value or sequence*, optional): batch-size candidates. The default is 64.
- ``valid_perc`` (*float*, optional): validation fraction. The default is 0.2.
- ``learning_rate`` (*value or sequence*, optional): fixed value or continuous lower and upper search bounds. The default bounds are ``(1e-5, 1e-4)``.
- ``dropout`` (*value or sequence*, optional): fixed value or continuous lower and upper search bounds. The default bounds are ``(0.1, 0.2)``.
- ``step_size`` (*value or sequence*, optional): scheduler-patience candidates. The default is ``(None,)``.
- ``architecture_y`` (*sequence of architectures*, optional): outcome-network candidates. The default is ``((1,),)``.
- ``architecture_z`` (*sequence of architectures*, optional): representation-network candidates. The default is ``((1024,), (2048,), (4096,))``.
- ``conv_layers`` (*list of dict*, optional): fixed convolutional front-end configuration.
- ``conv_activation`` (*callable*, optional): fixed convolutional activation factory. The default is ``torch.nn.ReLU``; use ``None`` to omit convolutional activations.
- ``bn`` (*value or sequence*, optional): batch-normalization candidates. The default is ``(False,)``.
- ``patience_min`` and ``patience_max`` (*int*, optional): inclusive early-stopping patience bounds. The defaults are 5 and 20.
- ``model_dir`` (*str*, optional): checkpoint directory used when refitting.
- ``random_state`` (*int*, optional): seed. The default is 42.
- ``verbose`` (*bool*, optional): whether to print model progress. The default is ``False``.

Native Python values and architectures are preferred. Legacy strings such as ``"100"`` and ``"[200, 1]"`` remain accepted.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.TarNet import TarNetHyperparameterTuner

   tuner = TarNetHyperparameterTuner(
       T=T,
       Y=Y,
       R=R,
       epoch=(100, 200),
       learning_rate=(1e-5, 1e-4),
       architecture_y=((200, 1), (100, 1)),
       architecture_z=((1024,), (2048,)),
   )

   study = tuner.tune(n_trials=100, refit=True)
   print(tuner.best_params_)
   best_model = tuner.best_model_

Methods
-------

- ``objective(trial)`` samples one configuration, trains it on the fixed split, and returns its best validation loss.
- ``tune(n_trials=50, timeout=None, study=None, sampler=None, pruner=None, n_jobs=1, refit=False, **optimize_kwargs)`` creates or uses a minimizing Optuna study, runs optimization, optionally refits, and returns the study.
- ``fit_best(study=None)`` fits and returns a :ref:`ref_TarNet` with the best resolved parameters.

``best_params_`` uses the :ref:`ref_TarNet` constructor names ``epochs`` and
``learning_rate`` rather than the tuner input name ``epoch`` or the
:ref:`ref_estimate_k_ate` names ``nepoch`` and ``lr``. It should not be
unpacked into ``estimate_k_ate`` without renaming those keys. Prefer
``fit_best`` when you need the selected ``TarNet``. If you construct it
manually from ``best_params_``, also pass the tuner's fixed ``conv_layers``,
``conv_activation``, ``min_delta=0``, ``verbose``, ``random_state``, and any
``model_dir``; these settings are not included in the dictionary.

Install version 0.2.1 with its ``tune`` extra as described in
:doc:`installation`. For causal estimation, tune on an independent sample or
within each outer training fold when possible. Use the default ``n_jobs=1``
for reproducible PyTorch trials; thread-parallel trials share random-number-
generator and accelerator state.
