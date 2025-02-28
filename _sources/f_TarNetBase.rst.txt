.. _ref_TarNetBase:

TarNetBase
===========

Description
-------------------------
The ``TarNetBase`` class implements the core neural network architecture used for treatment effect estimation. It consists of a shared representation network (deconfounder) and two outcome prediction networks—one for the untreated (control) and one for the treated group. The network supports optional dropout and batch normalization. The forward pass computes a latent representation from input data and then generates predictions for both treatment scenarios. Optionally, if ``return_prob`` is set to ``True``, the network applies a softmax activation to return predicted probabilities.

Parameters
----------
- **sizes_z** (*tuple*, optional): Sizes of the hidden layers for the shared representation model (deconfounder). Default is ``[2048]``.
- **sizes_y** (*tuple*, optional): Sizes of the hidden layers for the outcome prediction models (for both treated and control groups). Default is ``[200, 1]``.
- **dropout** (*float*, optional): Dropout rate applied to hidden layers. If not provided (``None``), dropout is not applied.
- **bn** (*bool*, optional): Whether to use batch normalization after each linear layer. Defaults to ``False``.
- **return_prob** (*bool*, optional): If ``True``, the model returns predicted probabilities (via softmax) for the outcomes; otherwise, raw outputs are returned. Defaults to ``False``.

Example Usage
-------------
.. code-block:: python

    from TNutil import TarNetBase

    # Initialize the TarNetBase model with custom parameters.
    model = TarNetBase(sizes_z=[2048], sizes_y=[200, 1], dropout=0.3, bn=True, return_prob=False)


Arguments:
-------
  - **sizes_z** (*tuple*, optional): Hidden layer sizes for deconfounder. Default: ``[2048]``.
  - **sizes_y** (*tuple*, optional): Hidden layer sizes for the outcome models. Default: ``[200, 1]``.
  - **dropout** (*float*, optional): Dropout rate to use (default: ``None``).
  - **bn** (*bool*, optional): Flag to enable batch normalization (default: ``False``).
  - **return_prob** (*bool*, optional): Flag to determine whether to return probabilities (default: ``False``).