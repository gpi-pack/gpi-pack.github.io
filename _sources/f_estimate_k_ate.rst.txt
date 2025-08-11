.. _ref_estimate_k_ate:

estimate_k_ate
===========

Description
-------
  Estimates the Average Treatment Effect (ATE) using the TarNet model with k-fold cross-fitting. For each fold, the model is trained on the training split, outcomes are predicted on the test split, the influence function (psi) is computed, and the mean ATE and its standard error are returned.

Arguments
-------
  - **R** (*list* or *np.ndarray*): Hidden state representations from a language model.
  - **Y** (*list* or *np.ndarray*): Outcome values.
  - **T** (*list* or *np.ndarray*): Treatment indicators.
  - **C** (*list* or *np.ndarray*, optional): Additional confounders.
  - **formula_C** (*str*, optional): Patsy-style formula for constructing confounders.
  - **data** (*pandas.DataFrame*, optional): DataFrame used with ``formula_C`` to build confounders.
  - **K** (*int*, optional): Number of folds for cross-fitting (default: 2).
  - **valid_perc** (*float*, optional): Fraction of training data for validation (default: 0.2).
  - **plot_propensity** (*bool*, optional): If True, plots the propensity score distribution (default: True).
  - **ps_model** (optional): Propensity score model (default: ``SpectralNormClassifier``).
  - **ps_model_params** (*dict*, optional): Hyperparameters for the propensity score model.
  - **batch_size** (*int*, optional): Batch size for TarNet training (default: 32).
  - **nepoch** (*int*, optional): Number of epochs to train TarNet (default: 200).
  - **step_size** (*int*, optional): Step size for the learning rate scheduler.
  - **lr** (*float*, optional): Learning rate for TarNet (default: 2e-5).
  - **dropout** (*float*, optional): Dropout rate for TarNet (default: 0.2).
  - **architecture_y** (*list*, optional): Architecture for the outcome model (default: [200, 1]).
  - **architecture_z** (*list*, optional): Architecture for the shared representation (default: [2048]).
  - **trim** (*list*, optional): Lower and upper bounds for trimming propensity scores (default: [0.01, 0.99]).
  - **bn** (*bool*, optional): Whether to use batch normalization (default: False).
  - **patience** (*int*, optional): Patience for early stopping (default: 5).
  - **min_delta** (*float*, optional): Minimum improvement for early stopping (default: 0).
  - **model_dir** (*str*, optional): Directory for saving model checkpoints.
  - **verbose** (*bool*, optional): If True, prints progress messages (default: True).

Returns
-------
  - **ate_est** (*float*): The estimated Average Treatment Effect.
  - **se_est** (*float*): The standard error (SE) of the ATE estimate.

Example Usage
-------

.. code-block:: python

    import pandas as pd
    from gpi_pack.TarNet import estimate_k_ate

    df = pd.DataFrame({
        'OutcomeVar': Y,
        'TreatmentVar': T,
        'conf1': np.random.rand(100),
        'conf2': np.random.rand(100)
    })

    ate, se = estimate_k_ate(
        R=R,
        Y=df['OutcomeVar'].values,
        T=df['TreatmentVar'].values,
        formula_C="conf1 + conf2",
        data=df,
        K=2
    )
    print("ATE:", ate, "SE:", se)
