.. _ref_mse_loss:

mse_loss
========

Description
-----------

The ``mse_loss`` function calculates the mean squared error between the true and predicted outcome tensors.

Arguments
---------

- ``y_true`` (*torch.Tensor*): observed outcomes.
- ``y_pred`` (*torch.Tensor*): predicted outcomes with a compatible shape.

Returns
-------

- *torch.Tensor*: scalar mean squared error.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.dyn_gpi import mse_loss

   loss = mse_loss(
       torch.tensor([1.0, 2.0]),
       torch.tensor([0.8, 2.1]),
   )
