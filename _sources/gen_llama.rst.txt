.. _generate_texts:

Generating Texts with LLaMa3
===========

For GPI, you need to generate texts and extract the internal representation of LLMs. This section describes how to generate texts and extract the internal representations using `LLaMa3 <https://huggingface.co/meta-llama>`_, which is one of the best open-source LLM.

.. note::
    For data generation, we recommend users to use GPUs. See :ref:`gpu_usage_section`.

How to use LLaMa3
---------
LLaMa3 is a large language model developed by Meta AI. It is designed to be efficient and effective for a wide range of natural language processing tasks. LLaMa3 is available in different sizes and versions. You can choose the one that best fits your needs and computational resources. Here, I use `Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_, which is the instruction-tuned version of LLaMa3 with 8 billion parameters.


.. note::

    To use Llama3, you need to have an access to the model (otherwise you will encounter the error "Cannot access gated repo for url"). You can request the access from `the model webpage <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_. Once you get access to the model, you also need to log in the huggingface and generate the access token. To generate access token, you need to click on "profile" button and select "Setting" and then click on "Access Token". See `the huggingface website <https://huggingface.co/docs/hub/en/security-tokens>`_ for more details about access token.

Creating Texts
---------

**gpi-pack** provides a function ``extract_and_save_hidden_states`` to generate texts using LLaMa3 and extract its internal representation. To use this function, you first need to load LLM and the corresponding tokenizer. When you use LLaMa3, you must need to supply your huggingface access token to the ``tokenizer``. Below is an example of how to load LLaMa3 and its tokenizer.

.. code-block:: python

    #loading required packages
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    ## Specify checkpoint (load LLaMa 3.1-8B)
    checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct' #model checkpoint of LLaMa3.1-8B-Instruct

    ## Load tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, token = <YOUR HUGGINGFACE TOKEN>)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.float16
    )

Suppose that you have the following list of prompts.

.. code-block:: python

    prompts = [
        'Create a biography of an American politician named Nathaniel C. Gilchrist',
        'Create a biography of an American politician named John Doe',
        'Create a biography of an American politician named Jane Smith',
        'Create a biography of an American politician named Mary Johnson',
        'Create a biography of an American politician named Robert Brown',
    ]

You can generate texts and extract the internal representation of LLaMa3 using the following code. You need to specify the directory to save the hidden states and the file name to save the generated texts.

.. code-block:: python

    from gpi_pack.llm import extract_and_save_hidden_states

    extract_and_save_hidden_states(
        prompts = prompts,
        output_hidden_dir = <YOUR HIDDEN DIR>, #directory to save hidden states
        save_name = <YOUR SAVE NAME>, #path and file name to save generated texts
        tokenizer = tokenizer,
        model = model,
        task_type = "create" # if you want to generate new texts, set task_type == "create"
    )

.. note::

    For GPI, it is required to use the deterministic decoding strategy. When you use ``extract_and_save_hidden_states``, the function internally sets the decoding strategy to be deterministic (greedy decoding).

Repeating Texts
---------

The function ``extract_and_save_hidden_states`` can also be used for the existing texts. To do so, you need to set ``task_type = "repeat"``.

.. code-block:: python

    from gpi_pack.llm import extract_and_save_hidden_states

    extract_and_save_hidden_states(
        prompts = prompts, #this text is the existing texts to be repeated
        output_hidden_dir = <YOUR HIDDEN DIR>, #directory to save hidden states
        save_name = <YOUR SAVE NAME>, #path and file name to save generated texts
        tokenizer = tokenizer,
        model = model,
        task_type = "repeat" # if you want to repeat existing texts, set task_type == "repeat"
    )

Arguments
---------

The function ``extract_and_save_hidden_states`` has the following arguments.

- ``prompts``: list of prompts to generate texts (**required**)
- ``output_hidden_dir``: directory to save the hidden states (**required**)
- ``save_name``: path and file name to save generated texts (**required**)
- ``tokenizer``: tokenizer of LLM (**required**)
- ``model``: pretrained LLM (**required**)
- ``task_type``: type of task. If you want to generate new texts, set ``task_type = "create"``. If you want to repeat existing texts, set ``task_type = "repeat"``. The default is "create". You can also provide string that specifies the system-level inputs (explained below).
- ``max_new_tokens``: maximum number of tokens to be generated. The default is 1000.
- ``prefix_hidden``: prefix of the hidden states files. The default is ``hidden_``.
- ``tokenizer_config``: configuration of tokenizer (optional)
- ``model_config``: configuration of model (optional)
- ``pooling``: pooling method to extract the internal representation. The default is "last". You can also use "mean" or "max".

System Prompt
---------
System prompt is a special type of prompt that is used to provide instructions or context to the LLM. The function ``extract_and_save_hidden_states`` instructs the task type (create or repeat) by using the system prompt. This function also allows you to specify your own system prompt by providing a string to ``task_type``. Below is an example of how to use the system prompt.

.. code-block:: python

    from gpi_pack.llm import extract_and_save_hidden_states

    extract_and_save_hidden_states(
        prompts = prompts,
        output_hidden_dir = <YOUR HIDDEN DIR>,
        save_name = <YOUR SAVE NAME>,
        tokenizer = tokenizer,
        model = model,
        #supply the user-specified system prompt
        task_type = "You are a text generator who always produces a biography of the instructed person"
    )