.. _ref_estimate_k_ate:

estimate_k_ate
==============

Description
-----------

The ``estimate_k_ate`` function estimates the Average Treatment Effect (ATE) with k-fold cross-fitting. Within each fold, it trains TarNet on the training observations, predicts the two potential outcomes on held-out observations, cross-fits a propensity model, and calculates doubly robust influence scores.

Arguments
---------

- ``R`` (*list* or *np.ndarray*): input representations.
- ``Y`` (*list* or *np.ndarray*): outcomes.
- ``T`` (*list* or *np.ndarray*): binary treatments.
- ``C`` (*array-like*, optional): observed confounder matrix.
- ``formula_C`` (*str*, optional): Patsy formula used with ``data`` to construct confounders.
- ``data`` (*pandas.DataFrame*, optional): data used by ``formula_C``.
- ``K`` (*int*, optional): number of cross-fitting folds. The default is 2.
- ``valid_perc`` (*float*, optional): TarNet validation fraction. The default is 0.2.
- ``plot_propensity`` (*bool*, optional): whether to plot the propensity distribution. The default is ``True``.
- ``ps_model`` (*class*, optional): propensity estimator class. The default is :ref:`ref_SpectralNormClassifier`.
- ``ps_model_params`` (*dict*, optional): propensity-model constructor arguments.
- ``batch_size`` (*int*, optional): TarNet batch size. The default is 32.
- ``nepoch`` (*int*, optional): TarNet epochs. The default is 200.
- ``step_size`` (*int*, optional): scheduler patience. ``None`` disables the scheduler.
- ``lr`` (*float*, optional): TarNet learning rate. The default is ``2e-5``.
- ``cluster`` (*list*, optional): cluster identifiers for the function's clustered-standard-error branch. In version 0.2.0, the cross-fitting implementation reorders influence scores without applying the same order to these identifiers; avoid this option unless you have verified the alignment for your workflow.
- ``dropout`` (*float*, optional): TarNet dropout. The default is 0.2.
- ``architecture_y`` (*list of int*, optional): outcome-network widths. The default is ``[200, 1]``.
- ``architecture_z`` (*list of int*, optional): representation-network widths. The default is ``[2048]``.
- ``conv_layers`` (*list of dict*, optional): convolutional front-end for image-shaped ``R``.
- ``conv_activation`` (*callable*, optional): convolutional activation factory. The default is ``torch.nn.ReLU``.
- ``trim`` (*list of float*, optional): lower and upper propensity bounds. The default is ``[0.01, 0.99]``.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``patience`` (*int*, optional): early-stopping patience. The default is 5.
- ``min_delta`` (*float*, optional): required validation-loss improvement. The default is 0.
- ``model_dir`` (*str*, optional): checkpoint directory.
- ``verbose`` (*bool*, optional): whether to print progress. The default is ``True``.

Returns
-------

- ``ate_est`` (*float*): estimated Average Treatment Effect.
- ``se_est`` (*float*): standard error. See the warning below before using ``cluster``.

.. warning::
   The clustered-standard-error branch in version 0.2.0 can misalign cluster identifiers with cross-fitted influence scores. Use the default unclustered standard error until the package preserves observation indices through cross-fitting, or independently verify and correct the ordering.

Example Usage
-------------

.. code-block:: python

   from gpi_pack import estimate_k_ate

   ate, se = estimate_k_ate(
       R=R,
       Y=df["OutcomeVar"].values,
       T=df["TreatmentVar"].values,
       formula_C="conf1 + conf2",
       data=df,
       K=2,
       architecture_y=[200, 1],
       architecture_z=[2048],
   )

   print("ATE:", ate, "SE:", se)
