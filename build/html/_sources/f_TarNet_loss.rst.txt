.. _ref_TarNet_loss:

TarNet_loss
===========

Purpose and Description
----------------------------
The ``TarNet_loss`` function calculates the loss for the TarNet model. Depending on whether the outcomes are categorical (indicated by ``return_probability=True``) or continuous, it computes either a negative log-likelihood loss or a mean squared error loss for the treated and control groups.

Arguments
---------
- **y_true** (*torch.Tensor*): The true outcome values. For categorical outcomes, these are class indices.
- **t_true** (*torch.Tensor*): The true treatment indicators (0 or 1).
- **y0_pred** (*torch.Tensor*): The predicted outcomes for the control group.
- **y1_pred** (*torch.Tensor*): The predicted outcomes for the treated group.

Returns
-------
- **loss** (*torch.Tensor*): A scalar tensor representing the combined loss.

Example Usage
-------
.. code-block:: python

    from TNutil import TarNet_loss

    loss = TarNet_loss(y_true, t_true, y0_pred, y1_pred)
    print("TarNet Loss:", loss.item())