.. _ref_dml_score:

dml_score
=========

Description
-----------
The ``dml_score`` function computes the augmented inverse-probability-weighted score contribution for every observation. The mean of these doubly robust contributions estimates the Average Treatment Effect (ATE). An influence function centered at the estimated ATE is obtained by subtracting that mean.

Arguments
---------
- **t** (*np.ndarray* or *torch.Tensor*): Binary treatment indicators.
- **y** (*np.ndarray* or *torch.Tensor*): Observed outcomes.
- **tpred** (*np.ndarray* or *torch.Tensor*): Estimated probabilities of treatment. Values should be strictly between zero and one; callers normally clip them before using this function.
- **ypred1** (*np.ndarray* or *torch.Tensor*): Predicted outcomes under treatment.
- **ypred0** (*np.ndarray* or *torch.Tensor*): Predicted outcomes under control.

All inputs are flattened internally and should contain the same number of observations. PyTorch inputs must be convertible to NumPy arrays; detach tensors that require gradients before calling the function.

Returns
-------
- **psi** (*np.ndarray*): One-dimensional array of doubly robust score contributions. Its mean is the ATE estimate.

Example Usage
-------------
.. code-block:: python

    from gpi_pack.TNutil import dml_score
    psi = dml_score(t, y, tpred, ypred1, ypred0)
    print("Influence function:", psi)
