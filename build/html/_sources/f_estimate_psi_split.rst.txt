.. _ref_estimate_psi_split:

estimate_psi_split
===========

Purpose and Description
----------------------------
The ``estimate_psi_split`` function estimates propensity scores from deconfounder using cross-fitting. It splits the data into two halves, trains a propensity model on each half, and then computes the influence function (psi) for each fold. The final psi values and the combined propensity scores are returned.

Arguments
---------
- **fr** (*np.ndarray* or *torch.Tensor*): The deconfounder.
- **t** (*np.ndarray* or *torch.Tensor*): Treatment indicators.
- **y** (*np.ndarray* or *torch.Tensor*): Observed outcomes.
- **y0** (*np.ndarray* or *torch.Tensor*): Predicted outcomes for the control group.
- **y1** (*np.ndarray* or *torch.Tensor*): Predicted outcomes for the treated group.
- **ps_model** (optional): The propensity score model class (default: ``SpectralNormClassifier``).
- **ps_model_params** (*dict*, optional): Hyperparameters for the propensity score model.
- **trim** (*list*, optional): Lower and upper quantile bounds for trimming propensity scores (default: ``[0.01, 0.99]``).
- **plot_propensity** (*bool*, optional): Whether to plot the propensity score distribution (default: ``False``).

Returns
-------
- **psi** (*np.ndarray*): The influence function values from both folds.
- **tpreds** (*np.ndarray*): The estimated propensity scores for all the samples.

Example Usage
-------
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
