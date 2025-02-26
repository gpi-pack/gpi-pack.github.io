.. _ref_extract_and_save_hiddens:

extract_and_save_hiddens
===========

Purpose and Description
----------------------------
The ``extract_and_save_hidden_states`` function provides a high-level interface for generating texts and extracting hidden representations from a language model. It takes a list of prompts, uses the specified tokenizer and model to generate texts, extracts hidden states using the chosen pooling strategy, saves the hidden states to disk, and then saves the generated texts to a pickle file.

Arguments
---------
- **prompts** (*list of str*): List of prompt strings.
- **output_hidden_dir** (*str*): Directory where the extracted hidden state files will be saved.
- **save_name** (*str*): Base filename (without extension) for saving the generated texts.
- **tokenizer** (*AutoTokenizer*): The tokenizer associated with the language model.
- **model** (*AutoModelForCausalLM*): The language model used for text generation.
- **task_type** (*str*, optional): The task type or instruction; default is ``"create"``.
- **max_new_tokens** (*int*, optional): Maximum number of tokens to generate (default: ``1000``).
- **prefix_hidden** (*str*, optional): Filename prefix for hidden states (default: ``"hidden_"``).
- **tokenizer_config** (*dict*, optional): Additional configuration for the tokenizer (default: empty dictionary).
- **model_config** (*dict*, optional): Additional configuration for the model (default: empty dictionary).
- **pooling** (*str*, optional): Pooling strategy to extract hidden states. Options are ``"last"`` (default) or ``"mean"``. Other values raise a ``ValueError``.

Returns
-------
- **None**: Saves the generated texts and hidden states to disk.

Example Usage
-------
.. code-block:: python

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from gpi_pack.llm import extract_and_save_hidden_states

    ## Specify checkpoint (load LLaMa 3.1-8B)
    checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct' #model checkpoint of LLaMa3.1-8B-Instruct

    ## Load tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = <YOUR HUGGINGFACE TOKEN>)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.float16
    )

    prompts = [
        'Create a biography of an American politician named Nathaniel C. Gilchrist',
        'Create a biography of an American politician named John Doe',
        'Create a biography of an American politician named Jane Smith',
        'Create a biography of an American politician named Mary Johnson',
        'Create a biography of an American politician named Robert Brown',
    ]

    extract_and_save_hidden_states(
        prompts = prompts,
        output_hidden_dir = <YOUR HIDDEN DIR>, #directory to save hidden states
        save_name = <YOUR SAVE NAME>, #path and file name to save generated texts
        tokenizer = tokenizer,
        model = model,
        task_type = "create" # if you want to generate new texts, set task_type == "create"
    )