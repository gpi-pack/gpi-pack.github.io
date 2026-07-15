.. _ref_estimate_psi_split:

estimate_psi_split
==================

Description
----------------------------
The ``estimate_psi_split`` function estimates propensity scores from deconfounder using cross-fitting. It splits the data into two halves, trains a propensity model on each half, and then computes the influence function (psi) for each fold. The final psi values and the combined propensity scores are returned.

Arguments
---------
- **fr** (*np.ndarray* or *torch.Tensor*): Estimated Deconfounders.
- **t** (*np.ndarray* or *torch.Tensor*): Treatment indicators.
- **y** (*np.ndarray* or *torch.Tensor*): Observed outcomes.
- **y0** (*np.ndarray* or *torch.Tensor*): Predicted outcomes for the control group.
- **y1** (*np.ndarray* or *torch.Tensor*): Predicted outcomes for the treated group.
- **ps_model** (optional): The propensity score model class (default: ``SpectralNormClassifier``).
- **ps_model_params** (*dict*, optional): Hyperparameters for the propensity score model.
- **trim** (*list*, optional): Absolute lower and upper bounds used to clip propensity scores (default: ``[0.01, 0.99]``).
- **plot_propensity** (*bool*, optional): Whether to plot the propensity score distribution (default: ``False``).

Returns
-------
- **psi** (*np.ndarray*): The influence function values from both folds.
- **tpreds** (*np.ndarray*): The estimated propensity scores for all the samples.

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
        plot_propensity=True
    )
