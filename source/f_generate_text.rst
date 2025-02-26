.. _ref_generate_text:

generate_text
===========

Description
----------------------------
The ``generate_text`` function generates texts from a list of prompts using a specified language model and tokenizer. It also extracts hidden states from the model’s outputs using a chosen pooling strategy and saves these hidden states to a designated directory. Finally, it returns the generated texts as a list.

Arguments
---------
- **tokenizer** (*AutoTokenizer*): The tokenizer used to process input text.
- **model** (*AutoModelForCausalLM*): The causal language model used for text generation.
- **instruction** (*str*): Instruction for the model (often obtained via ``get_instruction``).
- **prompts** (*list of str*): A list of prompts to generate text from.
- **max_new_tokens** (*int*): The maximum number of tokens to generate.
- **save_hidden** (*str*): Directory path where the extracted hidden states will be saved.
- **prefix_hidden** (*str*, optional): Filename prefix for saved hidden states (default: ``"hidden_"``).
- **tokenizer_config** (*dict*, optional): Additional configuration options for the tokenizer (default: empty dictionary).
- **model_config** (*dict*, optional): Additional configuration options for the model (default: empty dictionary).
- **pooling** (*str*, optional): Strategy for extracting hidden states. Options include:
  - ``"last"`` – use the last hidden state of the final token.
  - ``"mean"`` – compute the mean of the hidden states of all tokens.
  - Any other value raises a ``ValueError``.

Returns
-------
- **generated_texts** (*list of str*): A list of texts generated for each prompt.

Example Usage
-------
.. code-block:: python

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from gpi_pack.llm import generate_text

    # Load a tokenizer and model (e.g., GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    instruction = "You are a text generator who always produces the texts suggested by the prompts."
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