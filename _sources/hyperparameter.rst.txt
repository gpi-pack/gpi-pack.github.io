Hyperparameter Tuning
===========

If you use our representation learning method (TarNet), we recommend tuning the model's hyperparameters. Hyperparameters are parameters set prior to training (i.e., not learned during training) that can affect the model's performance. In this section, we explain how to tune the hyperparameters using **gpi_pack**.

Automated Hyperparameter Tuning
---------

For automated hyperparameter tuning, we use the framework called `Optuna <https://optuna.org/>`_. Optuna is a hyperparameter optimization framework designed for ease of use and efficiency. If you have not installed Optuna yet, please install it using the following command:

.. code-block:: bash

    pip install optuna

To use Optuna, you need to define an objective function to optimize and specify the set of hyperparameters to fine-tune. You can accomplish this using the `TarNetHyperparameterTuner`` class. This class is a wrapper around the Optuna framework and provides a simple interface for defining both the objective function and the hyperparameters. The following example demonstrates how to use the ``TarNetHyperparameterTuner`` class:

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

The following table lists the hyperparameters that can be tuned using the ``TarNetHyperparameterTuner`` class:


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