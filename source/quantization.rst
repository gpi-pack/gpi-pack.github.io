When an LLM Is Too Large
========================

A large language model can exceed GPU or system memory before generation
starts. Weight quantization reduces the memory used by linear-layer weights by
storing them at lower precision. It does not change the model's hidden width,
so the current **gpi_pack** LLM functions still require a model whose hidden
size is 4096.

This page uses 4-bit bitsandbytes quantization with
`Llama-3.1-8B-Instruct
<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_. Quantization can
change numerical outputs, so treat the checkpoint, quantization method, and
compute precision as part of the preprocessing specification and keep them
fixed across observations.

Installing bitsandbytes
-----------------------

Install compatible current versions of Transformers, Accelerate, and
bitsandbytes:

.. code-block:: bash

   python -m pip install --upgrade transformers accelerate bitsandbytes

Hardware support differs across bitsandbytes backends. Consult the
`Transformers bitsandbytes guide
<https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes>`_
for the current platform requirements.

Loading a 4-bit Model
---------------------

The following example uses Normal Float 4 (NF4), nested quantization, and an
automatic device map. You must first accept the Llama checkpoint's access
conditions and authenticate with Hugging Face.

.. code-block:: python

   import torch
   from transformers import (
       AutoModelForCausalLM,
       AutoTokenizer,
       BitsAndBytesConfig,
   )

   checkpoint = "meta-llama/Llama-3.1-8B-Instruct"

   if not torch.cuda.is_available():
       raise RuntimeError("This example expects a CUDA GPU.")

   compute_dtype = (
       torch.bfloat16
       if torch.cuda.is_bf16_supported()
       else torch.float16
   )

   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=compute_dtype,
       bnb_4bit_use_double_quant=True,
   )

   tokenizer = AutoTokenizer.from_pretrained(checkpoint)
   model = AutoModelForCausalLM.from_pretrained(
       checkpoint,
       quantization_config=quantization_config,
       device_map="auto",
       torch_dtype="auto",
   )

You can pass this ``tokenizer`` and ``model`` to
``extract_and_save_hidden_states`` exactly as in :ref:`generate_texts`.
``device_map="auto"`` is appropriate here because this is an inference
workflow; it can distribute model components across the available devices.

.. note::

   Quantization support is model- and hardware-dependent. It reduces weight
   memory but does not guarantee that a model will fit, because activations,
   the generation cache, and temporary tensors also consume memory.
