.. _ref_HyperparameterTuning:

Hyperparameter Tuning
=====================

If you use our representation learning method, we recommend tuning the neural-network hyperparameters. Hyperparameters are set before training and can affect both predictive performance and the learned deconfounder. **gpi_pack** provides Optuna-based tuners for the static TarNet and the Dynamic TarNet used for sequential data.

Automated Hyperparameter Tuning
-------------------------------

`Optuna <https://optuna.org/>`_ is a framework for automated hyperparameter optimization. Install the optional tuning dependencies with:

.. code-block:: bash

   python -m pip install --upgrade "gpi-pack[tune]"

The ``tune`` extra and the 0.2.1 tuner interfaces are available from PyPI.

For each trial, the **gpi_pack** tuners sample one configuration, fit the corresponding outcome model, and minimize its best held-out factual-outcome mean squared error. When the tuner creates the Optuna study, it uses a seeded TPE sampler and a median pruner by default.

Native Python values are recommended. For example, use ``(100, 200)`` for two epoch candidates and ``((16, 1), (32, 16, 1))`` for two outcome-network architectures. A single value fixes that hyperparameter instead of tuning it.

Static Hyperparameter Tuning
----------------------------

Use ``TarNetHyperparameterTuner`` for the static Text-as-Treatment, Text-as-Confounder, and Image-as-Treatment settings. Supply the same treatment, outcome, representation, and optional covariate arrays that you use for static estimation.

.. code-block:: python

   from gpi_pack.TarNet import TarNetHyperparameterTuner

   tuner = TarNetHyperparameterTuner(
       # Data
       T=df["TreatmentVar"].values,
       Y=df["OutcomeVar"].values,
       R=hidden_states,

       # Search space
       epoch=(100, 200),
       batch_size=(32, 64),
       learning_rate=(1e-5, 1e-4),
       dropout=(0.1, 0.3),
       step_size=(None, 5),
       architecture_y=((200, 1), (100, 1)),
       architecture_z=((1024,), (2048,)),
       bn=(False, True),
       patience_min=5,
       patience_max=20,
       random_state=42,
   )

   study = tuner.tune(n_trials=100, refit=True)

   print("Best validation loss:", tuner.best_score_)
   print("Best hyperparameters:", tuner.best_params_)
   best_model = tuner.best_model_

Setting ``refit=True`` fits ``best_model_`` with the selected configuration after tuning. You can instead call ``tuner.fit_best()`` later. This creates a fresh fit with the same validation design; it does not remove the validation split and train on every observation. The returned ``study`` is the Optuna study and can be used with Optuna's visualization and analysis tools.

Static Search Space
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 48 30
   :align: center

   * - Parameter
     - Interpretation
     - Example
   * - ``epoch``
     - Categorical candidates for the maximum number of training epochs.
     - ``(100, 200)``
   * - ``batch_size``
     - Categorical training batch-size candidates.
     - ``(32, 64)``
   * - ``learning_rate``
     - Log-scale continuous bounds. One value fixes the learning rate.
     - ``(1e-5, 1e-4)``
   * - ``dropout``
     - Continuous bounds for the dropout probability.
     - ``(0.1, 0.3)``
   * - ``step_size``
     - Categorical scheduler-patience candidates. ``None`` disables the scheduler.
     - ``(None, 5)``
   * - ``architecture_y``
     - Candidate outcome-network architectures. The final width is the outcome dimension.
     - ``((200, 1), (100, 1))``
   * - ``architecture_z``
     - Candidate deconfounder architectures. The final width is the learned representation dimension.
     - ``((1024,), (2048,))``
   * - ``bn``
     - Categorical batch-normalization choices.
     - ``(False, True)``
   * - ``patience_min``, ``patience_max``
     - Inclusive integer bounds for early-stopping patience.
     - ``5, 20``

The optional ``conv_layers`` and ``conv_activation`` arguments configure a fixed CNN front end for image representations; they are not sampled by Optuna. For the complete static API, see :ref:`ref_TarNetHyperparameterTuner`.

Dynamic Hyperparameter Tuning
-----------------------------

Use ``DynamicGPIHyperparameterTuner`` for sequential treatments, including the Video-as-Treatment setting. ``DynamicTarNetHyperparameterTuner`` is an alias of the same class. The dynamic tuner uses the following data:

- ``R``: vector or text representations with shape ``[N, T, F]`` or ``[F, N, T]``.
- ``W``: binary treatment histories with shape ``[N, T]``. Observed values are zero or one, and unused trailing positions are ``NaN``.
- ``Y``: one scalar outcome per unit with shape ``[N]`` or ``[N, 1]``.
- ``C``: optional static covariates with shape ``[N]`` or ``[N, P]``, or segment-varying covariates with shape ``[N, T]``, ``[N, T, P]``, or ``[P, N, T]``.
- ``R_video``: optional aligned video representations with shape ``[N, T, D, H, W]`` or ``[N, T, C_video, D, H, W]``.

Every unit must have at least one observed treatment. Representations and segment-varying covariates must be finite at observed positions; their padded values are replaced with zeros according to the mask inferred from ``W``. Static covariates must be finite for every unit and are not masked.

Supplying ``R_video`` automatically enables multimodal tuning. The tuner then samples the text and video encoder architectures in addition to the shared representation and outcome networks.

.. code-block:: python

   from gpi_pack import (
       DynamicGPIHyperparameterTuner,
       estimate_k_ipsi,
   )

   tuner = DynamicGPIHyperparameterTuner(
       # Sequential data
       R=R_text,
       R_video=R_video,  # omit for vector-only representations
       W=W,
       Y=df["OutcomeVar"].values,

       # Dynamic TarNet search space
       nepoch=(100, 200),
       batch_size=(16, 32),
       lr=(1e-5, 1e-3),
       dropout=(0.0, 0.3),
       architecture_y=((16, 1), (32, 16, 1)),
       architecture_z=((64, 32), (128, 64)),

       # Multimodal encoder search space
       text_hidden_dims=((512, 128), (1024, 256)),
       text_out_dim=(64, 128),
       video_channels=((8, 16), (8, 16, 32)),
       video_out_dim=(64, 128),

       valid_perc=0.2,
       random_state=42,
       device="auto",
   )

   study = tuner.tune(n_trials=50)

   print("Best validation loss:", tuner.best_score_)
   print("Best hyperparameters:", tuner.best_params_)

The resolved ``best_params_`` dictionary uses the same argument names as ``estimate_k_ipsi``. For multimodal data, it also contains the inferred ``text_input_dim`` and ``video_in_channels``. You can therefore pass it directly to the estimator:

.. code-block:: python

   result = estimate_k_ipsi(
       R=R_text,
       R_video=R_video,
       W=W,
       Y=df["OutcomeVar"].values,
       delta_seq=[0.5, 1.0, 2.0],
       K=5,
       **tuner.best_params_,
   )

If you need the fitted Dynamic TarNet outcome model itself, call ``tuner.fit_best()`` or set ``refit=True`` in ``tune``. This refit retains the tuner's validation split rather than training on every observation without validation:

.. code-block:: python

   best_model = tuner.fit_best()

Dynamic Search Space
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 48 30
   :align: center

   * - Parameter
     - Interpretation
     - Default search space
   * - ``nepoch``
     - Categorical candidates for the maximum number of Dynamic TarNet epochs.
     - ``(100, 200)``
   * - ``batch_size``
     - Categorical training batch-size candidates.
     - ``(16, 32)``
   * - ``lr``
     - Log-scale continuous learning-rate bounds. One value fixes the rate.
     - ``(1e-5, 1e-3)``
   * - ``dropout``
     - Continuous bounds for the dropout probability.
     - ``(0.0, 0.3)``
   * - ``architecture_y``
     - Candidate outcome-network architectures. Every candidate must end in 1 for the scalar outcome fitted in each model.
     - ``((16, 1), (32, 16, 1))``
   * - ``architecture_z``
     - Candidate per-segment deconfounder architectures.
     - ``((64, 32), (128, 64))``
   * - ``text_hidden_dims``
     - Candidate text-encoder hidden architectures used only in multimodal mode.
     - ``((1024, 256),)``
   * - ``text_out_dim``
     - Categorical encoded-text widths used only in multimodal mode.
     - ``(128,)``
   * - ``video_channels``
     - Candidate 3D-convolution channel architectures used only in multimodal mode.
     - ``((8, 16, 32),)``
   * - ``video_out_dim``
     - Categorical encoded-video widths used only in multimodal mode.
     - ``(128,)``

The fixed arguments ``valid_perc``, ``random_state``, ``device``, ``model_dir``, and ``verbose`` control validation, reproducibility, execution, and refitting but are not sampled. ``model_dir`` must already exist when it is used for the best-model refit. The tuner also fixes ``bn=False``, ``patience=5``, and ``min_delta=0``. In vector-only mode, the multimodal encoder options are not included in ``best_params_``.

For a ``video_channels`` candidate containing ``L`` convolutional layers, every input video depth and spatial dimension must be at least ``2 ** (L - 1)`` because the encoder pools between layers. For example, ``(8, 16, 32)`` requires these dimensions to be at least 4.

Repeated Outcomes
~~~~~~~~~~~~~~~~~

The dynamic tuner accepts one scalar outcome per unit. It does not accept the repeated-outcome array ``Y`` with shape ``[N, T]`` supported by :ref:`ref_estimate_k_ipsi`. To tune for a selected repeated-outcome segment ``s``, first retain eligible units and truncate every history through that segment:

.. code-block:: python

   import numpy as np

   s = 2  # zero-based outcome segment
   eligible = np.isfinite(W[:, s]) & np.isfinite(Y_repeated[:, s])

   segment_tuner = DynamicGPIHyperparameterTuner(
       R=R_text[eligible, : s + 1],
       R_video=R_video[eligible, : s + 1],
       W=W[eligible, : s + 1],
       Y=Y_repeated[eligible, s],
       nepoch=(100, 200),
       lr=(1e-5, 1e-3),
       architecture_y=((16, 1), (32, 16, 1)),
       architecture_z=((64, 32), (128, 64)),
   )

   segment_tuner.tune(n_trials=50)

You can tune eligible history prefixes separately or reuse a scalar-tuned configuration across the repeated-outcome estimator. One repeated-outcome call to ``estimate_k_ipsi`` uses one shared hyperparameter configuration for all outcome segments. In either case, keep the tuning design fixed and report it as part of the analysis.

Important Notes
---------------

- The tuners optimize held-out factual-outcome prediction, not the downstream propensity or backward-regression nuisance models and not the final causal estimate.
- For strict causal cross-fitting, tune on an independent sample or inside each outer training fold. Tuning globally on all outcomes before cross-fitting can leak outcome information into held-out estimates.
- Use the default ``n_jobs=1`` for reproducible PyTorch trials. Thread-parallel Optuna trials share random-number-generator and accelerator state; process-isolated workers are safer for parallel tuning.
- The validation split and model initialization seed are held fixed across trials so configurations are compared on the same split.

For the complete dynamic tuner API, see :ref:`ref_DynamicGPIHyperparameterTuner`.
