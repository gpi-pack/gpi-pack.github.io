.. _ref_TarNet:

TarNet
======

Description
-----------

The ``TarNet`` class is a training wrapper around :ref:`ref_TarNetBase`. It creates training and validation loaders, performs early stopping, optionally saves the best model state, and predicts outcomes and learned representations for user-supplied treatment values.

Parameters
----------

- ``epochs`` (*int*, optional): maximum training epochs. The default is 200.
- ``batch_size`` (*int*, optional): batch size. The default is 32.
- ``learning_rate`` (*float*, optional): AdamW learning rate. The default is ``2e-5``.
- ``architecture_y`` (*list of int*, optional): outcome-network widths. The default is ``[1]``.
- ``architecture_z`` (*list of int*, optional): representation-network widths. The default is ``[1024]``.
- ``conv_layers`` (*list of dict*, optional): optional convolutional front-end specifications for image-shaped inputs.
- ``conv_activation`` (*callable*, optional): convolutional activation factory. The default is ``torch.nn.ReLU``.
- ``dropout`` (*float*, optional): dropout probability. The default is 0.3.
- ``step_size`` (*int*, optional): reduce-on-plateau scheduler patience. ``None`` disables the scheduler.
- ``bn`` (*bool*, optional): whether to use batch normalization. The default is ``False``.
- ``patience`` (*int*, optional): early-stopping patience. The default is 5.
- ``min_delta`` (*float*, optional): required validation-loss improvement. The default is 0.01.
- ``model_dir`` (*str*, optional): directory where the best state is saved as ``best_TarNet.pth``.
- ``verbose`` (*bool*, optional): whether to print progress. The default is ``True``.
- ``random_state`` (*int*, optional): split and training seed. The default is 42.

Example Usage
-------------

.. code-block:: python

   import numpy as np
   from gpi_pack.TarNet import TarNet

   model = TarNet(
       architecture_y=[200, 1],
       architecture_z=[2048],
       epochs=100,
   )
   best_loss = model.fit(R, Y, T, valid_perc=0.2)

   y0, representation = model.predict(R, t=np.zeros(len(R)))
   y1, _ = model.predict(R, t=np.ones(len(R)))

Methods
-------

create_dataloaders
^^^^^^^^^^^^^^^^^^

``create_dataloaders(r_train, r_test, y_train, y_test, t_train, t_test, c_train=None, c_test=None)`` converts NumPy arrays when necessary and stores the training and validation loaders.

fit
^^^

``fit(R, Y, T, C=None, valid_perc=0.2, plot_loss=True, epoch_callback=None)`` trains the model with an internal validation split and returns the best validation loss as a float. ``epoch_callback`` is a keyword-only hook used by the tuner.

validate_step
^^^^^^^^^^^^^

``validate_step(use_confounder=False)`` calculates validation MSE and returns a scalar tensor.

predict
^^^^^^^

``predict(r, t, c=None, grad_required=False)`` predicts the outcome for the supplied treatment value and returns ``(y_preds, frs)``. ``y_preds`` has shape ``[N, architecture_y[-1]]``; without additional confounders, ``frs`` has shape ``[N, architecture_z[-1]]``. To obtain both potential outcomes, call the method once with an all-zero treatment vector and once with an all-one treatment vector, as in the example above.
