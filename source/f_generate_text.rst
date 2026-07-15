.. _ref_generate_text:

generate_text
=============

Description
----------------------------
The ``generate_text`` function generates texts from a list of prompts using a specified language model and tokenizer. It also extracts hidden states from the model’s outputs using a chosen pooling strategy and saves these hidden states to a designated directory. Finally, it returns the generated texts as a list.

Arguments
---------
- **tokenizer** (*AutoTokenizer*): The tokenizer used to process input text.
- **model** (*AutoModelForCausalLM*): The causal language model used for text generation. The current implementation reshapes hidden states to width 4096, so the model must use a compatible hidden size.
- **instruction** (*str*): Instruction for the model (often obtained via ``get_instruction``).
- **prompts** (*list of str*): A list of prompts to generate text from.
- **max_new_tokens** (*int*): The maximum number of tokens to generate.
- **save_hidden** (*str*): Existing directory where the extracted hidden states will be saved. This lower-level function does not create the directory.
- **prefix_hidden** (*str*, optional): Filename prefix for saved hidden states (default: ``"hidden_"``).
- **tokenizer_config** (*dict*, optional): Additional configuration options for the tokenizer (default: empty dictionary).
- **model_config** (*dict*, optional): Additional configuration options for the model (default: empty dictionary).
- **pooling** (*str*, optional): Strategy for extracting hidden states. Options include:
  - ``"last"`` – use the last-layer hidden state returned at the final generation step.
  - ``"mean"`` – average the last-layer states across generation steps after the initial step.
  - Any other value raises a ``ValueError``.

Returns
-------
- **generated_texts** (*list of str*): A list of texts generated for each prompt.

.. note::
    The current implementation requires a chat-template tokenizer and a model with hidden width 4096. Use ``extract_and_save_hidden_states`` when you want the output directory to be created automatically.

Example Usage
-------------
.. code-block:: python

    from pathlib import Path
    from gpi_pack.llm import generate_text, get_instruction

    # Load `tokenizer` and `model` as described in the LLaMa3 guide.
    Path("./hidden_states").mkdir(parents=True, exist_ok=True)
    instruction = get_instruction("create")
    prompts = ["Once upon a time", "In a galaxy far, far away"]
    texts = generate_text(
        tokenizer=tokenizer,
        model=model,
        instruction=instruction,
        prompts=prompts,
        max_new_tokens=50,
        save_hidden="./hidden_states",
        pooling="last"
    )

    for text in texts:
        print(text)
