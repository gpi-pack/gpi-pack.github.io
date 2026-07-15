.. _ref_generate_text:

generate_text
=============

Description
-----------

``generate_text`` applies one system instruction to a sequence of prompts. It
generates each response separately with non-sampling decoding, saves one
pooled last-layer tensor per prompt, and returns the decoded responses. With
``num_beams=1``, the usual model default, this is greedy decoding; a caller can
instead select non-sampling beam search through ``model_config``.

.. code-block:: text

   generate_text(
       tokenizer,
       model,
       instruction,
       prompts,
       max_new_tokens,
       save_hidden,
       prefix_hidden="hidden_",
       tokenizer_config={},
       model_config={},
       pooling="last",
   )

Arguments
---------

- ``tokenizer``: tokenizer with ``apply_chat_template`` and an ``eos_token_id``.
  The template result must be a mapping with ``input_ids`` and
  ``attention_mask`` and support ``.to(model.device)``.
- ``model``: causal language model with ``generate`` and ``device``. The
  current implementation reshapes every representation to width 4096, so the
  model's hidden size must be 4096.
- ``instruction`` (*str*): system-message content used for every prompt.
- ``prompts`` (*iterable of str*): prompts processed one at a time.
- ``max_new_tokens`` (*int*): maximum new tokens passed to ``model.generate``.
- ``save_hidden`` (*str*): existing output directory. This lower-level
  function does not create it.
- ``prefix_hidden`` (*str*, optional): exact prefix before the zero-based
  prompt index. The default ``"hidden_"`` produces ``hidden_0.pt``.
- ``tokenizer_config`` (*dict*, optional): additional chat-template keyword
  arguments.
- ``model_config`` (*dict*, optional): additional generation keyword
  arguments.
- ``pooling`` (*str*, optional): ``"last"`` or ``"mean"``. The default is
  ``"last"``.

The function mutates both supplied configuration dictionaries with
``dict.update``. It forces ``add_generation_prompt=False``, mapping and tensor
tokenizer output, and the following generation settings: ``max_new_tokens``,
``output_hidden_states=True``, ``return_dict_in_generate=True``,
``do_sample=False``, EOS padding, ``temperature=None``, and ``top_p=None``.
Those required values therefore override the same keys in user dictionaries.
The function does not override ``num_beams`` from ``model_config`` or the
model's generation configuration.

Pooling Behavior
----------------

- ``"last"`` selects the final layer at the final generation step and reshapes
  it to ``[-1, 4096]``.
- ``"mean"`` excludes the initial prompt-processing entry, stacks the final
  layer from each remaining generation step, reshapes to width 4096, and
  averages over rows. It fails if there are no remaining steps to stack.

The old exception text lists ``"all"``, but that option has no implementation.
Any value other than ``"last"`` or ``"mean"`` raises ``ValueError``.

Returns
-------

- *list of str*: decoded new tokens for each prompt, in input order.

The function also writes ``<save_hidden>/<prefix_hidden><index>.pt``. It does
not return the tensors and does not create the output directory.

Example Usage
-------------

.. code-block:: python

   from pathlib import Path
   from gpi_pack.llm import generate_text, get_instruction

   Path("outputs/hidden").mkdir(parents=True, exist_ok=True)

   texts = generate_text(
       tokenizer=tokenizer,
       model=model,
       instruction=get_instruction("create"),
       prompts=["Write a short biography of Ada Lovelace."],
       max_new_tokens=128,
       save_hidden="outputs/hidden",
       pooling="last",
   )
