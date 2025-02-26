.. _ref_load_hiddens:

load_hiddens
===========

Description
----------------------------
The ``load_hiddens`` function loads hidden state tensors from disk. Given a directory and a list of file identifiers (without the ``.pt`` extension), it reads each tensor file (optionally using a prefix), stacks them together, and returns a tensor of hidden representations.

Arguments
---------
- **directory** (*str*): The directory where the hidden state files are stored.
- **hidden_list** (*list*): List of file names (without the ``.pt`` extension) to be loaded.
- **prefix** (*str*, optional): A prefix to add to each file name (default: ``None``).
- **device** (*torch.device*, optional): The device on which to load the tensors (default: ``"cpu"``).

Returns
-------
- **tensors** (*torch.Tensor*): A tensor containing the stacked hidden representations.

Example Usage
-------
.. code-block:: python

    from gpi_pack.TNutil import load_hiddens

    # Assume hidden files are named like "hidden_last_1.pt", "hidden_last_2.pt", etc.
    hidden_list = [1, 2, 3]
    hidden_states = load_hiddens(directory="./hidden_states", hidden_list=hidden_list, prefix="hidden_last_")
    print("Loaded hidden states shape:", hidden_states.shape)