.. _ref_SpectralNormClassifier:

SpectralNormClassifier
===========

Purpose and Description
-------------------------
The ``SpectralNormClassifier`` class implements a feed-forward neural network for multi-class classification with spectral normalization. It also works for binary classification when ``num_classes`` is set to 2. This classifier applies spectral normalization to each linear layer in order to control the Lipschitz constant and improve training stability. Its architecture is a multi-layer perceptron (MLP) that can optionally include batch normalization and dropout in each hidden layer.

Parameters
----------
- **input_dim** (*int*): Number of input features in the data (dimension of X).
- **hidden_sizes** (*list of int*, optional): Sizes of the hidden layers. Defaults to ``[128, 64]``.
- **num_classes** (*int*, optional): Number of output classes. Defaults to ``2`` (binary classification).
- **n_power_iterations** (*int*, optional): Number of power iterations for computing the spectral norm in each layer. Defaults to ``1``.
- **dropout** (*float*, optional): Dropout probability for each layer. If ``0.0``, no dropout is applied. Defaults to ``0.0``.
- **batch_norm** (*bool*, optional): Whether to add a batch normalization layer after each linear layer. Defaults to ``False``.
- **lr** (*float*, optional): Learning rate for the Adam optimizer. Defaults to ``2e-6``.
- **nepoch** (*int*, optional): Maximum number of training epochs. Defaults to ``20``.
- **batch_size** (*int*, optional): Batch size used during training. Defaults to ``32``.
- **patience** (*int*, optional): Patience (in epochs) for early stopping on the validation set. Defaults to ``5``.
- **min_delta** (*float*, optional): Minimum improvement in validation loss required to reset patience. Defaults to ``1e-4``.
- **use_scheduler** (*bool*, optional): Whether to use a learning rate scheduler (e.g., StepLR or ReduceLROnPlateau). Defaults to ``False``.
- **scheduler_type** (*str*, optional): Scheduler type: ``"StepLR"`` or ``"ReduceLROnPlateau"``. Defaults to ``"ReduceLROnPlateau"``.
- **step_size** (*int*, optional): Step size for the scheduler. Interpreted differently depending on ``scheduler_type`` (default: ``5``).
- **gamma** (*float*, optional): Learning rate decay factor used by the scheduler. Defaults to ``0.5``.
- **valid_perc** (*float*, optional): Proportion of data to use for validation (train/valid split). Defaults to ``0.2``.

Example Usage
-------------
.. code-block:: python

    from gpi_pack.TNutil import SpectralNormClassifier

    # Initialize the classifier
    model = SpectralNormClassifier(input_dim=20, hidden_sizes=[64, 32], num_classes=2)

    # Train the classifier
    model.fit(deconfounder, T)

    # Predict class probabilities
    probs = model.predict_proba(deconfounder)
    print("Predicted probabilities:", probs)

    # Predict hard classes
    predictions = model.predict(deconfounder)
    print("Predicted classes:", predictions)

Methods
-------

fit
^^^
Purpose and Description:
  Trains the ``SpectralNormClassifier`` on the given dataset. It automatically splits the data into training and validation sets, performs training over a specified number of epochs, applies early stopping based on the validation loss, and optionally uses a learning rate scheduler.

Arguments:
  - **X** (*np.ndarray*): Input data of shape [n_samples, input_dim].
  - **y** (*np.ndarray*): Target class labels of shape [n_samples] (values in [0, num_classes-1]).

Example:

.. code-block:: python

    # Training the classifier
    model.fit(deconfounder_train, T_train)

predict_proba
^^^^^^^^^^^^^
Purpose and Description:
  Computes the class probabilities for each sample by performing a forward pass and then applying softmax to the output logits.

Arguments:
  - **X** (*np.ndarray*): Input data of shape [n_samples, input_dim].

Returns:
  - **probs** (*np.ndarray*): Array of shape [n_samples, num_classes] containing the predicted class probabilities.

Example:

.. code-block:: python

    probs = model.predict_proba(deconfounder_test)
    print("Predicted probabilities:", probs)

predict
^^^^^^^
Purpose and Description:
  Provides hard class predictions by selecting the class with the highest predicted probability for each sample.

Arguments:
  - **X** (*np.ndarray*): Input data of shape [n_samples, input_dim].

Returns:
  - **predictions** (*np.ndarray*): Array of shape [n_samples] with predicted class labels (ranging from 0 to num_classes-1).

Example:

.. code-block:: python

    predictions = model.predict(deconfounder_test)
    print("Predicted classes:", predictions)