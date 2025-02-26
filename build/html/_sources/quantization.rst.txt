When LLM is too big
===============

It is often the case that the LLM is too big to be loaded into the memory. In these cases, you encounter the error message ``CUDA out of memory``. This error message means that your GPU does not have enough memory to load the LLM.

In that case, you can use the quantization technique to reduce the size of the model. The quantization technique reduces the precision of the model parameters, which can significantly reduce the size of the model. The quantized model can be loaded into the memory and used for generating texts.

Model Quantization
----------

Here, we show how to use the quantization technique to reduce the size of the model (`LLaMa3 <https://huggingface.co/meta-llama>`_).

.. note::
    The quantization technique is not supported for all LLMs. You need to check the documentation of the LLM to see if the quantization technique is supported.

To use the quantization technique, you need to install the `bitsandbytes <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?bnb=4-bit>`_ package. You can install it using the following command:

.. code-block:: bash

    pip install bitsandbytes

Once you install the `bitsandbytes <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes?bnb=4-bit>`_ package, you can use the quantization technique to reduce the size of the model. Below is an example of how to use the quantization technique to reduce the size of LLaMa3.

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