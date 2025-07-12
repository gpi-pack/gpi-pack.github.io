.. _ref_ImageAsTreatment:

Image-As-Treatment
===========

Our framework **gpi_pack** also supports other modalities than texts, such as images. In this section, we describe how to use **gpi_pack** for image data, specifically focusing on the Image-as-Treatment setting. This setting is useful when you are interested in estimating the treatment effects of some image features while controlling the other confounding features. While image data has a unique characteristic, most of the estimation procedures are similar to Text-as-Treatment setting. Below, we describe how to use **gpi_pack** for Image-as-Treatment setting.

.. note::
    This part is based on our paper `GenAI-Powered Inference <https://arxiv.org/abs/2507.03897>`_. Please refer to the paper for the technical details.


How to estimate treatment effects
---------

**gpi_pack** offers the wrapper function ``estimate_k_ate`` to streamline the estimation of treatment effects using generative models' internal representations. This function handles all the necessary steps, including deconfounder estimation and propensity score estimation, so you only need to provide the data.

Suppose you have a DataFrame ``df`` containing the treatment variable, outcome variable, and texts, and you have already extracted the internal representations of the LLMs (saved as .pt files). Below is an example demonstrating how to use ``estimate_k_ate`` .

.. code-block:: python

    # loading required packages
    import pandas as pd

    # create a data frame
    df = pd.DataFrame({
        'TreatmentVar': [1, 0, 1, 0, 1],
        'OutcomeVar': [1, 0, 1, 0, 1],
        'Image_id': [0, 1, 2, 3, 4] # Image IDs corresponding to the internal representations you saved (e.g., "hidden_1.pt" for image indexed 1)
    })


Step 1: Load the Internal Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load the internal representations using the ``load_hiddens`` function:

.. code-block:: python

    # loading required packages
    from gpi_pack.TarNet import estimate_k_ate, load_hiddens

    # load hidden states stored as .pt files
    hidden_dir = <YOUR-DIRECTORY> # directory containing hidden states (e.g., "hidden_last_1.pt" for text indexed 1)

    hidden_states = load_hiddens(
        directory = hidden_dir,
        hidden_list= df['Image_id'].tolist(), # list of indices for hidden states
        prefix = "hidden_", # prefix of hidden states (e.g., "hidden_" for "hidden_1.pt")
    )

.. note::

    If you have not extracted internal representation, please refer to the section :ref:`generate_images`.

Note that in the case of images, the internal representations are typically not 1-dimensional like texts, so you need to flatten the internal representations before passing them to the ``estimate_k_ate`` function.

.. code-block:: python

    # flatten the hidden states
    hidden_states = [state.flatten() for state in hidden_states]

    # convert to numpy array
    hidden_states = np.array(hidden_states)



Step 2: Estimate the Treatment Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the internal representations are loaded, use ``estimate_k_ate`` to estimate the treatment effects:

.. code-block:: python

    # estimate treatment effects
    ate, se = estimate_k_ate(
        # Data (Inputs)
        R= hidden_states,
        Y= df['OutcomeVar'].values,
        T= df['TreatmentVar'].values,

        # Hyperparameters (optional)
        K=2, #K-fold cross-fitting
        lr = 2e-5, #learning rate
        architecture_y = [200, 1], #outcome model architecture
        architecture_z = [2048], #deconfounder architecture
    )


To compute a 95% confidence interval for the treatment effect estimate, use the following code:

.. code-block:: python

    # calculate 95% confidence interval
    lower_bound = ate - 1.96 * se
    upper_bound = ate + 1.96 * se

    print(f"ATE: {ate}, SE: {se}, 95% CI: ({lower_bound}, {upper_bound})")
    # ATE: 0.5, SE: 0.1, 95% CI: (0.3, 0.7)


For more details on the arguments of ``estimate_k_ate``, please refer to :ref:`ref_TextAsTreatment`