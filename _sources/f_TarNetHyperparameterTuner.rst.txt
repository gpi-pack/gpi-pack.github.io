.. _ref_TarNetHyperparameterTuner:

TarNetHyperparameterTuner
===========

Description
-------

The ``TarNetHyperparameterTuner`` class is used for hyperparameter tuning of the TarNet model. It leverages a hyperparameter optimization framework (e.g., Optuna) to explore different model configurations and determine the best set of hyperparameters based on validation loss.

Arguments
-------

  - **T**: Treatment variables.
  - **Y**: Outcome variables.
  - **R**: Internal representations.
  - **epoch** (*list of str*, optional): Epoch options as strings (default: ["100", "200"]).
  - **batch_size** (*int*, optional): Batch size for tuning (default: 64).
  - **valid_perc** (*float*, optional): Fraction of data for validation (default: 0.2).
  - **learning_rate** (*list of float*, optional): Range of learning rates (default: [1e-4, 1e-5]).
  - **dropout** (*list of float*, optional): Range of dropout rates (default: [0.1, 0.2]).
  - **step_size** (*list of int*, optional): List of step sizes (default: [5, 10]).
  - **architecture_y** (*list of list of str*, optional): Outcome model architecture options (default: ["[1]"]).
  - **architecture_z** (*list of list of str*, optional): Deconfounder architecture options (default: ["[1024]", "[2048]", "[4096]"]).
  - **bn** (*list of bool*, optional): Options for batch normalization (default: [True, False]).
  - **patience_min** (*int*, optional): Minimum patience value (default: 5).
  - **patience_max** (*int*, optional): Maximum patience value (default: 20).

Example Usage
-------

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

