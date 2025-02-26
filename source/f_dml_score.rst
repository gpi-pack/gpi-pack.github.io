.. _ref_dml_score:

dml_score
===========

Purpose and Description
----------------------------
The ``dml_score`` function computes the influence function for the average treatment effect (ATE) using a doubly robust estimation. It combines the predicted outcomes under treatment and control along with the observed treatment assignment to generate a score for each sample.

Arguments
---------
- **t** (*np.ndarray* or *torch.Tensor*): The treatment indicator.
- **y** (*np.ndarray* or *torch.Tensor*): The observed outcome.
- **tpred** (*np.ndarray* or *torch.Tensor*): The predicted treatment probability.
- **ypred1** (*np.ndarray* or *torch.Tensor*): The predicted outcome if treated.
- **ypred0** (*np.ndarray* or *torch.Tensor*): The predicted outcome if untreated.

Returns
-------
- **psi** (*np.ndarray*): The computed influence function for the ATE.

Example Usage
-------
.. code-block:: python

    from gpi_pack.TNutil import dml_score
    psi = dml_score(t, y, tpred, ypred1, ypred0)
    print("Influence function:", psi)