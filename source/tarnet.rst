.. _ref_TextAsTreatment:

Text-As-Treatment
===========

Text-As-Treatment is one of the most important settings where **gpi-pack** is useful. Instead of modeling the texts directly, we use the internal representation of the LLMs to estimate the treatment effects. By avoiding the direct modeling of the texts, we can achieve the accurate and more computationally efficient estimation of the treatment effects. This section provides the overview of the Text-As-Treatment setting and how to use **gpi-pack** to estimate the treatment effects using the internal representation of the LLMs.

.. note::
    This part is based on our paper `Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments <https://arxiv.org/abs/2410.00903>`_. Please refer to the paper for the technical details.

What is Text-As-Treatment?
---------
Text-As-Treatment refers to the setting where many texts are assigned to the participants and we are interested in how one specific feature (e.g., topics, sentiments) of texts influences the downstream outcomes. The key challenge in this setting is that texts contain some other featrues that might confound the relationship between the feature of interest and the outcome.

To address this challenge, `our paper <https://arxiv.org/abs/2410.00903>`_ proposes to use the internal representation of the LLMs to avoid the modeling of the texts, and we devised the representation learning method called **deconfounder**, by which we can estimate the treatment effects of the feature of interest without directly observing all the confounding features. Deconfounder is estimated using `TarNet`, which inputs the internal representation and predict the outcome under the treatment and control conditions using the shared **deconfounder**. The following figure shows the architecture of `TarNet`.

.. image:: /_static/images/tarnet.png
   :alt: TarNet architecture
   :width: 600px

Once we estimate the **deconfoudner** and the outcome models, we estimate the propensity score model based on the estimated deconfounder. Finally, we use the estimated outcome models and the propensity score model to estimate the treatment effects using the double machine learning techniques.

How to estimate treatment effects
---------

**gpi_pack** provides the wrapper function ``estimate_k_ate`` to estimate the treatment effects using the internal representation of the LLMs. This function internally handles all the estimation procedures including deconfounder estimation and propensity score estimations, and you only need to specify the data. Below is the example of how to use ``estimate_k_ate`` to estimate the treatment effects.

Suppose that we have the following data frame ``df`` that contains the treatment variable, outcome variable, and the texts, and we already extracted the internal representation of the LLMs and saved them as .pt files.

.. code-block:: python

    # loading required packages
    import pandas as pd

    # create a data frame
    df = pd.DataFrame({
        'TreatmentVar': [1, 0, 1, 0, 1],
        'OutcomeVar': [1, 0, 1, 0, 1],
        'Text': [
            'Create a biography of an American politician named Nathaniel C. Gilchrist',
            'Create a biography of an American politician named John Doe',
            'Create a biography of an American politician named Jane Smith',
            'Create a biography of an American politician named Mary Johnson',
            'Create a biography of an American politician named Robert Brown',
        ]
    })

.. note::

    If you have not extracted internal representation, please refer to the section :ref:`generate_texts`.

Firstly, you need to load the internal representation of the LLMs. You can use the function ``load_hiddens`` to load the internal representation of the LLMs. The following is an example of how to use ``load_hiddens`` to load the internal representation of the LLMs.

.. code-block:: python

    # loading required packages
    from gpi_pack.TarNet import estimate_k_ate, load_hiddens

    # load hidden states stored as .pt files
    hidden_dir = <YOUR-DIRECTORY> # directory containing hidden states (e.g., "hidden_last_1.pt" for text indexed 1)

    hidden_states = load_hiddens(
        directory = hidden_dir,
        hidden_list= df.index.tolist(), # list of indices for hidden states
        prefix = "hidden_last_", # prefix of hidden states (e.g., "hidden_last_" for "hidden_last_1.pt")
    )

Once you load the internal representation of the LLMs, you can use the function ``estimate_k_ate`` to estimate the treatment effects. The following is an example of how to use ``estimate_k_ate`` to estimate the treatment effects.

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

Theoretically, the estimator follows the normal distribution when the sample size is large enough. Therefore, we can use the following code to calculate the 95% confidence interval of the treatment effects.

.. code-block:: python

    # calculate 95% confidence interval
    lower_bound = ate - 1.96 * se
    upper_bound = ate + 1.96 * se

    print(f"ATE: {ate}, SE: {se}, 95% CI: ({lower_bound}, {upper_bound})")
    # ATE: 0.5, SE: 0.1, 95% CI: (0.3, 0.7)

How to control confounders
---------

Sometimes, we want to control the confounders that are not included in the texts. In this case, we can use the function ``estimate_k_ate`` to estimate the treatment effects while controlling the confounders. The following is an example of how to use ``estimate_k_ate`` to estimate the treatment effects while controlling the confounders.

There are two ways to control the confounders. One is to use the formula and the ``pandas`` data frame. To do this, your data frame must contain the confounders as columns, and you need to specify the formula of confounders (``formula_c = "conf1 + conf2"``) and the data frame (``data = df``) in the function ``estimate_k_ate``.

.. code-block:: python

    # Method 1: supply covariates with a formula and DataFrame
    ate, se = estimate_k_ate(
        R= hidden_states,
        Y= df['OutcomeVar'].values,
        T= df['TreatmentVar'].values,
        formula_c="conf1 + conf2",
        data=df,
        K=2, #K-fold cross-fitting
        lr = 2e-5, #learning rate
        #Outcome model architecture
        # [100, 1] means that the deconfounder is passed to the intermediate layer with size 100,
        # and then it passes to the output layer with size 1.

        #Outcome model architecture
        # [100, 1] means that the deconfounder is passed to the intermediate layer with size 100,
        # and then it passes to the output layer with size 1.
        architecture_y = [200, 1],

        #Deconfounder model architecture:
        # [2048] means that the input (hidden states) is passed to the intermediate layer with size 2048.
        # The size of last layer (last number in the list) corresponds to the dimension of the deconfounder.
        architecture_z = [2048],
    )

The other way is to use the design matrix of confounders. To do this, you need to create a design matrix of confounders and specify it in the ``C`` argument.

.. code-block:: python

    # Method 2: supply covariates using a design matrix
    import numpy as np #load numpy module
    C_mat = np.column_stack([df['conf1'].values, df['conf2'].values])

    ate, se = estimate_k_ate(
        R= hidden_states,
        Y= df['OutcomeVar'].values,
        T= df['TreatmentVar'].values,
        C=C_mat, #design matrix of confounding variable
        K=2, #K-fold cross-fitting
        lr = 2e-5, #learning rate

        #Outcome model architecture
        # [100, 1] means that the deconfounder is passed to the intermediate layer with size 100,
        # and then it passes to the output layer with size 1.
        architecture_y = [200, 1],

        #Deconfounder model architecture:
        # [2048] means that the input (hidden states) is passed to the intermediate layer with size 2048.
        # The size of last layer (last number in the list) corresponds to the dimension of the deconfounder.
        architecture_z = [2048],
    )

Visualizing Propensity Scores
---------

For Text-As-Treatment, we need to assume that textual feature and confounding features are disentangled. This assumption is called **separability**, and this can be directly diagnosed by visualizing the propensity scores. For this reason, we recommend users to visualize the propensity scores to check if the separability is not violated. If the propensity score shows the extreme values (0 or 1), it is likely that some confounding features are entangled with the treatment feature of interest.

Our function ``estimate_k_ate`` provides the option to visualize the propensity scores. To visualize the propensity scores, you need to set ``plot_propensity = True`` in the function ``estimate_k_ate`` (which is the default option). The following is an example of how to visualize the propensity scores.

.. code-block:: python

    # estimate treatment effects
    ate, se = estimate_k_ate(
        R= hidden_states,
        Y= df['OutcomeVar'].values,
        T= df['TreatmentVar'].values,
        K=2, #K-fold cross-fitting
        lr = 2e-5, #learning rate
        architecture_y = [200, 1], #outcome model architecture
        architecture_z = [2048], #deconfounder architecture
        plot_propensity = True, #visualize propensity scores
    )

.. image:: /_static/images/propensity.png
   :alt: propensity score
   :width: 600px

Hyperparameters
---------

The function ``estimate_k_ate`` has the following parameters.

- ``R``: list or np.ndarray
    A list or NumPy array of hidden states extracted from LLM. Shape: (N, d_R) where N is the number of samples and d_R is the dimension of hidden states. You can load the stored hidden states using `load_hiddens` function.
- ``Y``: list or np.ndarray
    A list or NumPy array of outcomes, shape: (N,).
- ``T``: list or np.ndarray
    A list or NumPy array of treatments, shape: (N,). Typically binary (0 or 1).
- ``C``: list or np.ndarray, optional
    A matrix of additional confounders, shape: (N, d_C). If provided, these will be concatenated to R along axis=1. You can pass either this parameter directly or use `formula_c` and `data`.
- ``formula_c``: str, optional
    A Patsy-style formula (e.g., `"conf1 + conf2"`) that specifies how to build the confounder matrix from a DataFrame. If this is provided, `data` must also be provided, and `C` will be constructed via `dmatrix(formula_c, data)`. Intercept is removed from the design matrix.
- ``data``: pandas.DataFrame, optional
    The DataFrame containing the columns used in `formula_c`. If `formula_c` is set, this parameter is required. The resulting design matrix is then concatenated to R as additional confounders.
- ``K``: int, default=2
    Number of cross-fitting folds (K-fold split).
- ``valid_perc``: float, default=0.2
    Proportion of the training set to use for validation when fitting TarNet in each fold.
- ``plot_propensity``: bool, default=True
    Whether to plot the propensity score distribution in the console or a graphing interface (implementation-specific).
- ``ps_model``: object, optional
    A model/classifier used to estimate the propensity score. By default, we use a neural network with Spectral Normalization (to ensure Lipshitz continuity).
- ``ps_model_params``: dict, optional
    Hyperparameters for `ps_model`. For example, `{"input_dim": 2048}` if using a custom model requiring an input dimension.
- ``batch_size``: int, default=32
    Batch size for TarNet training.
- ``nepoch``: int, default=200
    Number of epochs to train TarNet.
- ``step_size``: int, optional
    Step size for the learning rate scheduler (if applicable).
- ``lr``: float, default=2e-5
    Learning rate for TarNet.
- ``dropout``: float, default=0.2
    Dropout rate for TarNet layers.
- ``architecture_y``: list, default=[200, 1]
    List specifying the layer sizes for the outcome heads (treatment-specific networks or final layers). For example, [200, 1] means that the outcome model has two hidden layers, the first with 200 units and the second with 1 unit.
- ``architecture_z``: list, default=[2048]
    List specifying the layer sizes for the deconfounder. For example, [2048, 2048] means that the deconfounder has two hidden layers, each with 2048 units.
- ``trim``: list, default=[0.01, 0.99]
    Trimming bounds for the propensity score. Propensity scores outside this range will be replaced with the nearest bound.
- ``bn``: bool, default=False
    Whether to apply batch normalization in TarNet.
- ``patience``: int, default=5
    Patience for early stopping in TarNet training (number of epochs without improvement).
- ``min_delta``: float, default=0
    Minimum improvement threshold for early stopping.
- ``model_dir``: str, optional
    Directory path where the model checkpoints might be saved. If provided, the best model will be saved here and loaded for predictions.
- ``verbose``: bool, default=True
    Whether to print additional information during training.