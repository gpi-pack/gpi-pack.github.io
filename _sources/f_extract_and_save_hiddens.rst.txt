.. _ref_extract_and_save_hidden_states:

extract_and_save_hidden_states
==============================

Description
-----------

``extract_and_save_hidden_states`` is the high-level local LLM workflow. It
chooses a built-in or custom system instruction, calls
:ref:`ref_generate_text`, and writes the generated responses with their
original prompts.

.. code-block:: text

   extract_and_save_hidden_states(
       prompts,
       output_hidden_dir,
       save_name,
       tokenizer,
       model,
       task_type="create",
       max_new_tokens=1000,
       prefix_hidden="hidden_",
       tokenizer_config={},
       model_config={},
       pooling="last",
   )

Arguments
---------

- ``prompts`` (*list of str*): prompts or existing texts to process.
- ``output_hidden_dir`` (*str*): directory for tensor files. It is created when
  missing.
- ``save_name`` (*str*): pickle path without the ``.pkl`` suffix. The function
  does not create this path's parent directory.
- ``tokenizer``: chat-template tokenizer accepted by
  :ref:`ref_generate_text`.
- ``model``: causal language model whose hidden width is 4096 in the current
  implementation.
- ``task_type`` (*str*, optional): ``"create"``, ``"repeat"``, or a custom
  system instruction. The default is ``"create"``.
- ``max_new_tokens`` (*int*, optional): per-prompt generation limit. The
  default is 1000.
- ``prefix_hidden`` (*str*, optional): tensor filename prefix. The default is
  ``"hidden_"``.
- ``tokenizer_config`` and ``model_config`` (*dict*, optional): extra options
  passed to the lower-level function. They are mutated in place, and required
  deterministic-generation values override conflicting keys.
- ``pooling`` (*str*, optional): ``"last"`` or ``"mean"``. The default is
  ``"last"``. ``"all"`` is not implemented despite an old docstring and
  exception message that mention it.

Returns
-------

The function returns ``None``. It writes one tensor named
``<prefix_hidden><index>.pt`` per prompt and writes ``<save_name>.pkl`` with
columns ``X`` (generated text) and ``P`` (original prompt).

``task_type="repeat"`` asks the model to repeat each input, but the function
does not verify exact reproduction. Check the saved ``X`` and ``P`` values
before analysis.

Example Usage
-------------

.. code-block:: python

   from pathlib import Path
   from gpi_pack.llm import extract_and_save_hidden_states

   Path("outputs").mkdir(parents=True, exist_ok=True)

   extract_and_save_hidden_states(
       prompts=prompts,
       output_hidden_dir="outputs/hidden",
       save_name="outputs/generated_texts",
       tokenizer=tokenizer,
       model=model,
       task_type="repeat",
       max_new_tokens=256,
       pooling="last",
   )
