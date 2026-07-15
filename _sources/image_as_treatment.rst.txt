.. _ref_ImageAsTreatment:

Image-As-Treatment
==================

Our framework **gpi_pack** also supports other modalities than texts, such as images. In this section, we describe how to use **gpi_pack** for image data, specifically focusing on the Image-as-Treatment setting. This setting is useful when you are interested in estimating the treatment effects of some image features while controlling the other confounding features. While image data has a unique characteristic, most of the estimation procedures are similar to Text-as-Treatment setting. Below, we describe how to use **gpi_pack** for Image-as-Treatment setting.

.. note::
    This part is based on our paper `GenAI-Powered Inference <https://arxiv.org/abs/2507.03897>`_. Please refer to the paper for the technical details.


How to estimate treatment effects
---------------------------------

**gpi_pack** offers the wrapper function ``estimate_k_ate`` to streamline the estimation of treatment effects using generative models' internal representations. This function handles all the necessary steps, including deconfounder estimation and propensity score estimation, so you only need to provide the data.

Suppose you have a DataFrame ``df`` containing the treatment variable, outcome variable, and image IDs, and you have already extracted the internal representations of an image model (saved as ``.pt`` files). Below is an example demonstrating how to use ``estimate_k_ate``.

.. code-block:: python

    # loading required packages
    import pandas as pd

    # create a data frame
    df = pd.DataFrame({
        'TreatmentVar': [1, 0, 1, 0, 1],
        'OutcomeVar': [1, 0, 1, 0, 1],
        'Image_id': [0, 1, 2, 3, 4] # Image IDs corresponding to the internal representations you saved (e.g., "hidden_1.pt" for image indexed 1)
    })

This five-row data frame illustrates the required organization only. Neural-network fitting and causal estimation require a substantively adequate sample.


Step 1: Load the Internal Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load the internal representations using the ``load_hiddens`` function:

.. code-block:: python

    # loading required packages
    from gpi_pack import estimate_k_ate, load_hiddens

    # load hidden states stored as .pt files
    hidden_dir = "outputs/image_hidden" # directory containing files such as "hidden_1.pt"

    hidden_states = load_hiddens(
        directory = hidden_dir,
        hidden_list= df['Image_id'].tolist(), # list of indices for hidden states
        prefix = "hidden_", # prefix of hidden states (e.g., "hidden_" for "hidden_1.pt")
    )

.. note::

    If you have not extracted the internal representations, please refer to :ref:`generate_images`.

Unlike text representations, image representations often retain spatial dimensions. **gpi_pack** provides two ways to use them. You can flatten each representation and use the default feed-forward network, or preserve the spatial structure and add a convolutional neural network (CNN).


Using Flattened Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest approach is to flatten each image representation before passing it to the ``estimate_k_ate`` function.

.. code-block:: python

    # flatten the hidden states
    hidden_states_flat = hidden_states.reshape(hidden_states.shape[0], -1)


Using a Convolutional Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CNN can be useful when the location of features within the image representation is important. Instead of flattening the representations in advance, the CNN learns local spatial patterns before passing its output to the deconfounder and treatment-conditioned outcome network.

For this approach, ``hidden_states`` must have the shape ``(N, C, H, W)``, where ``N`` is the number of observations, ``C`` is the number of channels, and ``H`` and ``W`` are the height and width. Then, define the convolutional layers as follows:

.. code-block:: python

    # keep the spatial structure of the image representations
    # load_hiddens removes a singleton second dimension; restore a
    # one-channel image axis when each saved tensor was [1, H, W].
    if hidden_states.ndim == 3:
        hidden_states = hidden_states[:, None, :, :]
    assert hidden_states.ndim == 4

    conv_layers = [
        {
            "in_channels": hidden_states.shape[1],
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "pool": {"type": "max", "kernel_size": 2},
        },
        {
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "pool": {"type": "max", "kernel_size": 2},
        },
    ]

The first layer must specify ``in_channels``. Each layer must specify
``out_channels``. The current implementation recognizes exactly
``kernel_size``, ``stride``, ``padding``, ``dilation``, ``groups``, and
``bias`` as additional ``torch.nn.Conv2d`` options; other keys such as
``padding_mode`` are ignored. The optional ``pool`` dictionary adds max
pooling; set ``"type": "avg"`` to use average pooling. Set
``"spectral_norm": True`` on a layer to apply spectral normalization. By
default, **gpi_pack** applies a ReLU activation after each convolutional layer
and flattens the final feature maps internally. Pass ``conv_activation=None``
to omit the convolutional activations.



Step 2: Estimate the Treatment Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the internal representations are loaded, use ``estimate_k_ate`` to estimate the treatment effects:

.. code-block:: python

    # estimate treatment effects
    ate, se = estimate_k_ate(
        # Data (Inputs)
        R= hidden_states_flat,
        Y= df['OutcomeVar'].values,
        T= df['TreatmentVar'].values,

        # Hyperparameters (optional)
        K=2, #K-fold cross-fitting
        lr = 2e-5, #learning rate
        architecture_y = [200, 1], #outcome model architecture
        architecture_z = [2048], #deconfounder architecture
    )


To use the CNN defined above, pass the unflattened image representations and ``conv_layers`` to the same function:

.. code-block:: python

    # estimate treatment effects using a CNN
    ate, se = estimate_k_ate(
        # Data (Inputs)
        R=hidden_states,
        Y=df['OutcomeVar'].values,
        T=df['TreatmentVar'].values,

        # CNN and other hyperparameters (optional)
        conv_layers=conv_layers,
        K=2,
        lr=2e-5,
        architecture_y=[200, 1],
        architecture_z=[2048],
    )

.. note::

    All image representations in ``R`` must have the same shape. The CNN configuration is a hyperparameter choice, so you should adjust the number of layers, channels, and pooling operations for the size of your representations and validate the choices on held-out data.


To compute a 95% confidence interval for the treatment effect estimate, use the following code:

.. code-block:: python

    # calculate 95% confidence interval
    lower_bound = ate - 1.96 * se
    upper_bound = ate + 1.96 * se

    print(f"ATE: {ate}, SE: {se}, 95% CI: ({lower_bound}, {upper_bound})")


For more details on the arguments of ``estimate_k_ate``, please refer to :ref:`ref_estimate_k_ate`.
