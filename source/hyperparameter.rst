Hyperparameter Tuning
===========

If you use our representation learning method (TarNet), we recommend you to tune the hyperparameters of the model. The hyperparameters are the parameters that are not learned during training, but are set before training. These parameters can have a significant impact on the performance of the model, and tuning changes the performance of **deconfounder**. In this section, we will introduce how to tune the hyperparameters of the model using **gpi-pack**.

Automated Hyperparameter Tuning
---------

For the automated hyperparameter tuning, we use the existing framework called `Optuna <https://optuna.org/>`_. `Optuna <https://optuna.org/>`_ is a hyperparameter optimization framework that is designed to be easy to use and flexible. If you have not installed `Optuna <https://optuna.org/>`_, please install it using the following command:

.. code-block:: bash

    pip install optuna

To use `Optuna <https://optuna.org/>`_, we need to define the objective function that we want to optimize and the set of hyperparameters we want to fine-tune. You can use the class ``TarNetHyperparameterTuner`` to do this. The ``TarNetHyperparameterTuner`` class is a wrapper around the `Optuna <https://optuna.org/>`_ framework, and it provides a simple interface for defining the objective function and the hyperparameters. The following is an example of how to use the ``TarNetHyperparameterTuner`` class:

.. code-block:: python

    from gpi_pack.TarNet import TarNetHyperparameterTuner
    import optuna

    # Load data and set hyperparameters
    obj = TarNetHyperparameterTuner(
        # Data
        T = df['TreatmentVar'].values, 
        Y = df['OutcomeVar'].values, 
        R = hidden_states, 

        # Hyperparameters
        epoch = ["100", "200"], #try either 100 epochs or 200 epochs
        learning_rate = [1e-4, 1e-5], #draw learning rate in the range (1e-4, 1e-5)
        dropout = [0.1, 0.2], #draw dropout rate in the range (1e-4, 1e-5)

        # Outcome model architecture:
        # [100, 1] means that the deconfounder is passed to the intermediate layer with size 100,
        # and then it passes to the output layer with size 1.
        architecture_y = ["[200, 1]", "[100,1]"], #either [200, 1] or [100, 1] (size of layers)

        #Deconfounder model architecture:
        # [1024] means that the input (hidden states) is passed to the intermediate layer with size 1024.
        # The size of last layer (last number in the list) corresponds to the dimension of the deconfounder.
        architecture_z = ["[1024]", "[2048]"] #either [1024] or [2048]
    )

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(obj.objective, n_trials=100) #runs 100 trials to seek the best hyperparameter

    #Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)


List of Hyperparameters
---------

The following is the list of hyperparameters that can be tuned using the ``TarNetHyperparameterTuner`` class:


.. list-table::
   :header-rows: 1
   :widths: 25 40 35
   :align: center

   * - Parameter Name
     - Description
     - Tuner Input Example
   * - ``epoch``
     - Number of epochs (pick up the best one from the list)
     - ``epoch = ["100", "200"]``
   * - ``learning_rate``
     - Range of learning rate (uniformly draw the best ones from the range)
     - ``learning_rate = [1e-4, 1e-5]``
   * - ``dropout``
     - Range of dropout (uniformly draw the best ones from the range)
     - ``dropout = [0.1, 0.2]``
   * - ``step_size``
     - List of step sizes for learning rate scheduler
     - ``step_size = [100, 200]``
   * - ``architecture_y``
     - List of architectures for outcome models (pick up the best one from the list)
     - ``architecture_y = ["[256, 128, 1]", "[128, 64, 1]"]``
   * - ``architecture_z``
     - List of architectures for deconfoudner (pick up the best one from the list)
     - ``architecture_z = ["[2048]", "[4096, 2048]"]``
   * - ``bn``
     - List of batch normalization options (pick up the best one from the list)
     - ``bn = [True, False]``
   * - ``patience``
     - The number of epochs to wait for improvement before stopping training
     - ``patience_max = 20, patience_min = 10``