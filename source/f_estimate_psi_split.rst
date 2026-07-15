.. _ref_estimate_psi_split:

estimate_psi_split
==================

Description
-----------
The ``estimate_psi_split`` function estimates propensity scores from the deconfounder using two-fold cross-fitting. It splits row indices with ``random_state=42``, trains a propensity model on each half, predicts the opposite half, and computes doubly robust score contributions. It also prints the average held-out treatment-classification accuracy.

Arguments
---------
- **fr** (*np.ndarray* or *torch.Tensor*): Estimated deconfounders with shape ``[N, F]``.
- **t** (*np.ndarray* or *torch.Tensor*): Binary treatment indicators.
- **y** (*np.ndarray* or *torch.Tensor*): Observed outcomes.
- **y0** (*np.ndarray* or *torch.Tensor*): Predicted outcomes for the control group.
- **y1** (*np.ndarray* or *torch.Tensor*): Predicted outcomes for the treated group.
- **ps_model** (*class*, optional): Propensity-model class implementing ``fit`` and ``predict_proba``. The probability of class 1 is used. The default is :ref:`ref_SpectralNormClassifier`.
- **ps_model_params** (*dict*, optional): Constructor arguments for the propensity score model. Direct use of the default classifier requires ``{"input_dim": fr.shape[1]}``.
- **trim** (*list* or *None*, optional): Absolute lower and upper bounds used to clip propensity scores (default: ``[0.01, 0.99]``). Use ``None`` to disable clipping.
- **plot_propensity** (*bool*, optional): Whether to plot the propensity score distribution (default: ``False``).

Returns
-------
- **psi** (*np.ndarray*): Doubly robust score contributions from both folds.
- **tpreds** (*np.ndarray*): Cross-fitted propensity scores in the original input row order, clipped unless ``trim=None``.

The propensity scores are restored to the original row order. The influence scores are concatenated in held-out fold order, so users calling this lower-level function directly should not assume that ``psi`` follows the original row order.

Example Usage
-------------
.. code-block:: python


    from gpi_pack.TNutil import estimate_psi_split

    psi, tpreds = estimate_psi_split(
        fr = deconfounder,
        t = treatment,
        y = outcome,
        y0 = y0_pred,
        y1 = y1_pred,
        ps_model_params={"input_dim": deconfounder.shape[1]},
        plot_propensity=True
    )
