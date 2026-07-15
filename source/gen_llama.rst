.. _generate_texts:

Generating Texts with Llama 3
==============================

For text GPI, you generate or regenerate one text at a time and save a
fixed-width representation from the language model. This page uses
`Llama-3.1-8B-Instruct
<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_, whose hidden width
of 4096 matches the current implementation in **gpi_pack**.

.. note::

   The package reshapes every saved LLM representation to width 4096. Changing
   only the checkpoint does not make the high-level function compatible with
   a model that has a different hidden width. See :doc:`gen_llm` for a
   model-independent manual workflow.

Loading Llama 3.1
-----------------

Llama-3.1-8B-Instruct requires Transformers 4.43 or later and is a gated Meta
checkpoint. If an existing environment contains an older Transformers release,
upgrade it before loading the model:

.. code-block:: bash

   python -m pip install --upgrade "transformers>=4.43" accelerate

Accept the checkpoint's access conditions
on the model page, create a Hugging Face token with permission to read gated
models, and make the token available without writing it directly into your
script. For example, log in once with ``hf auth login`` or set the ``HF_TOKEN``
environment variable.

The following example loads the model on the available accelerator devices:

.. code-block:: python

   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer

   checkpoint = "meta-llama/Llama-3.1-8B-Instruct"

   compute_dtype = (
       torch.bfloat16
       if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
       else (torch.float16 if torch.cuda.is_available() else "auto")
   )

   tokenizer = AutoTokenizer.from_pretrained(checkpoint)
   model = AutoModelForCausalLM.from_pretrained(
       checkpoint,
       device_map="auto",
       torch_dtype=compute_dtype,
   )

The example uses ``torch_dtype="auto"`` on a CPU-only machine. The
``torch_dtype`` spelling is compatible with the documented Transformers 4.43
minimum; newer versions also accept ``dtype``. Loading this checkpoint on CPU
is possible but generally slow and requires substantial system memory. See
:ref:`gpu_usage_section` and :doc:`quantization` for other options.

Creating Texts
--------------

Suppose you have one prompt for each observation:

.. code-block:: python

   prompts = [
       "Create a biography of an American politician named Nathaniel C. Gilchrist",
       "Create a biography of an American politician named John Doe",
       "Create a biography of an American politician named Jane Smith",
   ]

Use ``task_type="create"`` to apply the package's built-in text-creation system
instruction:

.. code-block:: python

   from gpi_pack.llm import extract_and_save_hidden_states

   extract_and_save_hidden_states(
       prompts=prompts,
       output_hidden_dir="outputs/hidden",
       save_name="outputs/generated_texts",
       tokenizer=tokenizer,
       model=model,
       task_type="create",
       max_new_tokens=256,
       pooling="last",
   )

The function processes prompts separately and forces non-sampling decoding
(``do_sample=False``). With the model's usual ``num_beams=1`` setting this is
greedy decoding, but ``model_config`` can select beam search. It creates
``outputs/hidden`` and writes
``hidden_0.pt``, ``hidden_1.pt``, and so on. It also writes
``outputs/generated_texts.pkl``, whose columns ``X`` and ``P`` contain the
generated texts and original prompts. The parent directory of ``save_name``
must already exist.

The package also forces ``add_generation_prompt=False`` when applying the chat
template. This differs from the current Llama-3.1-Instruct model-card example,
which uses ``True`` to append the assistant header. The package setting cannot
be changed through ``tokenizer_config`` because it overwrites that key; record
this formatting choice when comparing representations with an external Llama
workflow.

Repeating Existing Texts
------------------------

Use ``task_type="repeat"`` when each element of ``prompts`` is an existing text
that the model should regenerate:

.. code-block:: python

   extract_and_save_hidden_states(
       prompts=existing_texts,
       output_hidden_dir="outputs/repeated_hidden",
       save_name="outputs/repeated_texts",
       tokenizer=tokenizer,
       model=model,
       task_type="repeat",
       max_new_tokens=256,
       pooling="last",
   )

The instruction asks the model to repeat the input, but language-model output
is not guaranteed to be character-for-character identical. Inspect the saved
``X`` and ``P`` columns before analysis. The representation is taken from the
model's generation states, not directly from the input token embeddings.

Pooling
-------

The current implementation supports two pooling values:

- ``"last"`` saves the last layer's state returned at the final generation
  step. This is the default.
- ``"mean"`` averages the last-layer states across generation steps after the
  initial prompt-processing step. It requires the generation to contain
  enough steps for that list to be nonempty.

Both options save a tensor with width 4096. ``"all"`` is mentioned in an old
package error message and docstring but is not implemented; any value other
than ``"last"`` or ``"mean"`` raises ``ValueError``.

Custom System Instruction
-------------------------

Any ``task_type`` other than the exact strings ``"create"`` and ``"repeat"``
is used verbatim as the system instruction:

.. code-block:: python

   extract_and_save_hidden_states(
       prompts=prompts,
       output_hidden_dir="outputs/hidden",
       save_name="outputs/generated_texts",
       tokenizer=tokenizer,
       model=model,
       task_type=(
           "You are a text generator who always produces a biography "
           "of the instructed person."
       ),
   )

Additional Arguments
--------------------

``prefix_hidden`` changes the prefix before the zero-based prompt index.
``tokenizer_config`` and ``model_config`` pass additional keyword arguments to
the chat template and generation call. The package overrides its required
generation settings, including ``do_sample=False``, hidden-state output,
return type, padding token, and ``max_new_tokens``. It does not override
``num_beams``. The supplied dictionaries are mutated in place, so pass fresh
dictionaries rather than reusing them across different configurations.

For the complete signature and return behavior, see
:ref:`ref_extract_and_save_hidden_states`.
