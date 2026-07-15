Generating Texts without a Local GPU
=====================================

`NNsight <https://nnsight.net/>`_ can execute model traces through the National
Deep Inference Fabric (NDIF). The model weights remain on the remote service,
and only values marked for saving are returned. This makes it possible to
generate text and extract internal states without a local GPU.

NNsight and NDIF are third-party services, not **gpi_pack** components. Their
available models, quotas, API, and access policies can change, so check the
`NNsight remote-execution guide
<https://nnsight.net/features/13_remote_execution/>`_ before running a large
job.

Installing and Authenticating
------------------------------

Install NNsight and obtain an NDIF API key from the account page linked in the
official guide:

.. code-block:: bash

   python -m pip install --upgrade nnsight

The preferred options are to set ``NDIF_API_KEY`` in your environment or save
the key through NNsight's configuration interface:

.. code-block:: python

   import os
   from nnsight import CONFIG

   CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

Large gated models can also require a Hugging Face token so NNsight can load
their tokenizer and configuration. Accept the checkpoint's conditions and set
``HF_TOKEN`` or authenticate with the Hugging Face CLI.

Choosing an Available Model
---------------------------

NDIF deployment status changes over time. Inspect the status before selecting
a checkpoint:

.. code-block:: python

   from nnsight import ndif

   print(ndif.status())

The example below uses a Llama instruction checkpoint. Replace the identifier
with the exact deployed model key shown by NDIF when necessary. NNsight creates
a lightweight local model skeleton on the ``meta`` device; it does not download
the remote weights.

.. code-block:: python

   from nnsight import LanguageModel

   checkpoint = "meta-llama/Llama-3.1-70B-Instruct"
   model = LanguageModel(checkpoint)
   tokenizer = model.tokenizer

Remote Generation and Hidden States
-----------------------------------

The following Llama-specific example uses the current NNsight generation API.
It processes each prompt independently and performs non-sampling generation
(``do_sample=False``), which is greedy with the selected checkpoint's default
``num_beams=1``. It collects the last Llama decoder layer at every generation
step. Other architectures use different module paths; inspect the wrapped
model before replacing the checkpoint.

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
       "You are a text generator who always produces the text requested "
       "by the user."
   )
   max_new_tokens = 256
   pooling = "last"  # or "mean"

   save_hidden = Path("outputs/nnsight_hidden")
   save_hidden.mkdir(parents=True, exist_ok=True)
   generated_texts = []

   for k, prompt in enumerate(prompts):
       messages = [
           {"role": "system", "content": instruction},
           {"role": "user", "content": prompt},
       ]
       input_ids = tokenizer.apply_chat_template(
           messages,
           add_generation_prompt=True,
           tokenize=True,
           return_tensors="pt",
       )

       with model.generate(
           input_ids,
           max_new_tokens=max_new_tokens,
           do_sample=False,
           remote=True,
       ) as tracer:
           hidden_steps = list().save()

           # A bounded iterator allows the code after the loop to execute.
           for _ in tracer.iter[:max_new_tokens]:
               hidden_steps.append(
                   model.model.layers[-1].output[0][
                       :, -1, :
                   ].detach().float().cpu()
               )

           output_ids = model.generator.output.save()

       generated_ids = output_ids[0, input_ids.shape[-1]:]
       generated_texts.append(
           tokenizer.decode(generated_ids, skip_special_tokens=True)
       )

       if pooling == "last":
           representation = hidden_steps[-1]
       elif pooling == "mean":
           generation_steps = hidden_steps[1:]
           if not generation_steps:
               raise RuntimeError(
                   "Mean pooling requires more than one generation step."
               )
           representation = torch.stack(generation_steps).mean(dim=0)
       else:
           raise ValueError("pooling must be 'last' or 'mean'")

       torch.save(
           representation.float().cpu(),
           save_hidden / f"hidden_{k}.pt",
       )

   pd.DataFrame({"X": generated_texts, "P": prompts}).to_pickle(
       "outputs/nnsight_generated.pkl"
   )

At the first generation step, the model processes the full prompt; later steps
normally contain one new token position. Selecting ``[:, -1, :]`` produces one
vector per step. The ``"mean"`` rule excludes the first prompt-processing
entry, matching the rule used by the package's local LLM workflow.

Only objects connected to ``.save()`` are downloaded from NDIF. This example
saves the full list of selected layer vectors because both pooling choices are
shown. For a large production run, reduce the values remotely before saving
when possible to lower transfer and storage costs.

Important Notes
---------------

- ``extract_and_save_hidden_states`` expects a local Transformers model and
  cannot be passed a remote NNsight wrapper. The manually saved ``.pt`` files
  can still be loaded by :ref:`ref_load_hiddens`.
- Remote execution sends prompts and traced computations to NDIF. Review its
  data-handling terms before submitting sensitive material.
- Record the checkpoint revision, NNsight version, module path, pooling rule,
  generation settings, and execution date. Model deployments and remote
  environments can change independently of this package.
