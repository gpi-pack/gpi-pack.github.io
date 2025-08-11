.. _ref_TarNet:

TarNet
===========

Description
----------------------------
The ``TarNet`` class is a wrapper for a treatment effect estimation model using a shared representation (deconfounder) and treatment-specific outcome models. It integrates the TarNetBase model, data loading, training, validation, prediction, and evaluation functionality. The class also supports saving and loading model checkpoints.

Parameters
-------
  - **epochs** (*int*, optional): Number of training epochs (default: 200).
  - **batch_size** (*int*, optional): Batch size for training (default: 32).
  - **learning_rate** (*float*, optional): Learning rate for the optimizer (default: 2e-5).
  - **architecture_y** (*list*, optional): Layer sizes for the outcome model (default: [1]).
  - **architecture_z** (*list*, optional): Layer sizes for the shared representation (default: [1024]).
  - **dropout** (*float*, optional): Dropout rate (default: 0.3).
  - **step_size** (*int*, optional): Step size for the learning rate scheduler (default: None).
  - **bn** (*bool*, optional): Whether to use batch normalization (default: False).
  - **patience** (*int*, optional): Number of epochs with no improvement before early stopping (default: 5).
  - **min_delta** (*float*, optional): Minimum improvement threshold for early stopping (default: 0.01).
  - **model_dir** (*str*, optional): Directory to save model checkpoints (default: None).
  - **return_probablity** (*bool*, optional): If True, model outputs probabilities (default: False).
  - **verbose** (*bool*, optional): If True, prints additional information during training (default: True).

Example Usage
-------

.. code-block:: python

    from TarNet import TarNet

    model = TarNet(
        epochs=100,
        batch_size=32,
        learning_rate=1e-4,
        architecture_y=[200, 1],
        architecture_z=[2048],
        dropout=0.2,
        bn=True,
        model_dir="./model_checkpoint"
    )
    model.fit(R, Y, T, valid_perc=0.2, plot_loss=True)
    y0_preds, y1_preds, frs = model.predict(R)

Methods
-------

fit
^^^
Purpose and Description:
  Trains the TarNet model using internal representations (R), outcomes (Y), and treatment indicators (T). It performs a train/validation split, trains the model with early stopping and optional learning rate scheduling, and optionally plots the training and validation loss curves.

Arguments:
  - **R** (*np.ndarray* or *torch.Tensor*): Internal representations for all samples.
  - **Y** (*np.ndarray* or *torch.Tensor*): Outcome values.
  - **T** (*np.ndarray* or *torch.Tensor*): Treatment indicators.
  - **valid_perc** (*float*, optional): Fraction of the data used for validation.
  - **plot_loss** (*bool*, optional): If True, plots the loss curves (default: True).

Example:

.. code-block:: python

    model = TarNet(epochs=50)
    model.fit(R, Y, T, valid_perc=0.2, plot_loss=True)


predict
^^^^^^^
Purpose and Description:
  Processes internal representation data in batches and returns predictions. It outputs the predicted outcomes for the control (T=0) and treated (T=1) groups, as well as the latent representation extracted by the model.

Arguments:
  - **r** (*np.ndarray* or *torch.Tensor*): Internal representation data for prediction.

Returns:
  - **y0_preds** (*torch.Tensor*): Predicted outcomes for the control group.
  - **y1_preds** (*torch.Tensor*): Predicted outcomes for the treated group.
  - **frs** (*torch.Tensor*): Deconfounder extracted by the model.

Example:

.. code-block:: python

    y0, y1, fr = model.predict(R)
    print("Control predictions:", y0)
    print("Treated predictions:", y1)
