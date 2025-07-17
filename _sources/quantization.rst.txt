When LLM is too big
===============

Sometimes a large language model (LLM) cannot be loaded into memory, resulting in a ``CUDA out of memory``. This error indicates that your GPU lacks sufficient memory to load the LLM.

In such cases, you can use quantization techniques to reduce the model's size. Quantization lowers the precision of the model parameters, significantly reducing its size. As a result, the quantized model can be loaded into memory and used for generating text.

Model Quantization
----------

Below is an example of how to use quantization to reduce the size of the model (using `LLaMa3 <https://huggingface.co/meta-llama>`_).

.. note::
    Quantization is not supported for all LLMs. Please consult the LLM's documentation to determine whether it supports this technique.

To use quantization, first install the `bitsandbytes <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?bnb=4-bit>`_ package with the following command

.. code-block:: bash

    pip install bitsandbytes

After installing the package, you can apply quantization to reduce the size of LLaMa3. The example below demonstrates how to do this:

.. code-block:: python

    #loading required packages
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    ## Specify checkpoint (load LLaMa 3.1-8B)
    checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct' #model checkpoint of LLaMa3.1-8B-Instruct

    ## Load tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        token = <YOUR HUGGINGFACE TOKEN>
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True # load the model in 4-bit
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_compute_dtype=torch.bfloat16, # computation precision
        bnb_4bit_use_double_quant=True, # use double quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        quantization_config=quantization_config,
        device_map="auto",
    )