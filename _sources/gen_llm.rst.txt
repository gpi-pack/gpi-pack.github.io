Generating Texts with Other LLMs
=================================

The high-level ``extract_and_save_hidden_states`` function currently reshapes
representations to width 4096. A checkpoint with another hidden width therefore
requires a small manual generation loop. This page uses the open-weight
`Gemma-2-2B instruction-tuned checkpoint
<https://huggingface.co/google/gemma-2-2b-it>`_ as an example.

The same pattern can be adapted to another Transformers causal language model,
but first check its model card for access conditions, chat-template roles,
precision, and generation requirements. A model having
``output_hidden_states`` does not by itself guarantee the same statistical
meaning as another model's representation, so keep the checkpoint and
extraction rule fixed across observations.

Loading Gemma 2
---------------

Accept the Gemma usage license on Hugging Face before loading the files. The
following example lets Transformers choose the checkpoint's configured
precision and distribute the model over available devices:

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer

   checkpoint = "google/gemma-2-2b-it"
   tokenizer = AutoTokenizer.from_pretrained(checkpoint)
   model = AutoModelForCausalLM.from_pretrained(
       checkpoint,
       device_map="auto",
       torch_dtype="auto",
   )

Gemma's instruction chat template uses ``user`` and ``model`` roles rather
than a separate ``system`` role. The example therefore places the task
instruction and observation-specific prompt in one user message.

Generating and Saving Representations
--------------------------------------

Process prompts separately so that the generation for one observation cannot
depend on other observations in the same batch:

.. code-block:: python

   from pathlib import Path

   import pandas as pd
   import torch

   prompts = [
       "Create a biography of an American politician named Nathaniel C. Gilchrist",
       "Create a biography of an American politician named John Doe",
       "Create a biography of an American politician named Jane Smith",
   ]
   instruction = (
       "Create the text requested below. Return only the requested text."
   )

   save_hidden = Path("outputs/gemma_hidden")
   save_hidden.mkdir(parents=True, exist_ok=True)
   generated_texts = []

   for k, prompt in enumerate(prompts):
       messages = [
           {
               "role": "user",
               "content": f"{instruction}\n\n{prompt}",
           }
       ]
       inputs = tokenizer.apply_chat_template(
           messages,
           add_generation_prompt=True,
           tokenize=True,
           return_dict=True,
           return_tensors="pt",
       ).to(model.device)

       with torch.inference_mode():
           outputs = model.generate(
               **inputs,
               max_new_tokens=256,
               do_sample=False,
               num_beams=1,
               pad_token_id=tokenizer.eos_token_id,
               output_hidden_states=True,
               return_dict_in_generate=True,
           )

       generated_ids = outputs.sequences[
           0, inputs["input_ids"].shape[-1]:
       ]
       generated_texts.append(
           tokenizer.decode(generated_ids, skip_special_tokens=True)
       )

       # Last layer, last token position, final generation step.
       representation = outputs.hidden_states[-1][-1][:, -1, :]
       torch.save(
           representation.float().cpu(),
           save_hidden / f"hidden_{k}.pt",
       )

   pd.DataFrame({"X": generated_texts, "P": prompts}).to_pickle(
       "outputs/gemma_generated.pkl"
   )

``outputs.hidden_states`` is organized first by generation step and then by
model layer. The expression ``[-1][-1][:, -1, :]`` selects the final step,
final layer, and final token position while retaining the batch dimension. It
does not hard-code the hidden width.

Mean Pooling Across Generation Steps
------------------------------------

To mirror the package's ``pooling="mean"`` rule, replace the representation
line with:

.. code-block:: python

   generation_steps = outputs.hidden_states[1:]
   if not generation_steps:
       raise RuntimeError("Mean pooling requires more than one generation step.")

   representation = torch.stack(
       [step[-1][:, -1, :] for step in generation_steps]
   ).mean(dim=0)

The first entry is excluded because it represents the initial forward pass
over the full prompt. Save the resulting tensor using the same ``float().cpu()``
conversion shown above.

.. note::

   These examples use greedy decoding for reproducibility. Deterministic
   decoding does not guarantee identical results across different model,
   library, precision, device, or kernel versions. Record that environment
   together with the checkpoint revision and extraction rule.
