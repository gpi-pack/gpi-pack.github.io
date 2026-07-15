.. _ref_extract_and_save_hidden_states:

extract_and_save_hidden_states
==============================

Description
-----------

The ``extract_and_save_hidden_states`` function is the high-level LLM workflow. It selects a built-in or custom system instruction, deterministically generates one response for every prompt, saves one pooled hidden-state tensor per prompt, and saves the responses and original prompts to a pickle file.

Arguments
---------

- ``prompts`` (*list of str*): prompts or texts to process.
- ``output_hidden_dir`` (*str*): directory for hidden-state ``.pt`` files.
- ``save_name`` (*str*): pickle path without the ``.pkl`` extension.
- ``tokenizer`` (*transformers tokenizer*): tokenizer with chat-template support.
- ``model`` (*causal language model*): model used for deterministic generation. Its hidden width must be 4096 in the current implementation.
- ``task_type`` (*str*, optional): ``"create"``, ``"repeat"``, or a custom system instruction. The default is ``"create"``.
- ``max_new_tokens`` (*int*, optional): maximum generated tokens. The default is 1000.
- ``prefix_hidden`` (*str*, optional): hidden-state filename prefix. The default is ``"hidden_"``.
- ``tokenizer_config`` (*dict*, optional): additional chat-template options.
- ``model_config`` (*dict*, optional): additional generation options.
- ``pooling`` (*str*, optional): ``"last"`` uses the final generation step's last-layer state; ``"mean"`` averages last-layer states across generation steps after the initial step. The default is ``"last"``.

Returns
-------

The function writes the hidden states and ``<save_name>.pkl`` and returns ``None``.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.llm import extract_and_save_hidden_states

   extract_and_save_hidden_states(
       prompts=prompts,
       output_hidden_dir="outputs/hidden",
       save_name="outputs/generated_texts",
       tokenizer=tokenizer,
       model=model,
       task_type="repeat",
       pooling="last",
   )
