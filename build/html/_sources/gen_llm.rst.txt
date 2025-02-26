Generating Texts with Other LLMs
===========

.. note::
    For data generation, we recommend users to use GPUs. See :ref:`gpu_usage_section` for how to use GPUs.

Sometimes, you may want to use other LLMs than LLaMa3. While the function ``extract_and_save_hidden_states`` can be compatible with some other LLM by changing the checkpoint, this function is not guaranteed to work with all LLMs. In this section, we will show how to use other LLMs.

Below, we use `Gemma2 <https://huggingface.co/google/gemma-2-2b-it>`_ as an example. Gemma2 is a large language model developed by Google. It is designed to be efficient and effective for a wide range of natural language processing tasks.

.. note::
    You need to use open-source LLMs to use GPI. You can pick up your favorite LLMs from the list of open-source LLMs, and `Huggingface <https://huggingface.co/>`_ has a list of open-source LLMs for various tasks.

Example: Gemma2
---------

Firstly, you load the LLM and its tokenizer. You can use the following code to load Gemma2.

.. code-block:: python

    # loading required packages
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    ## Specify checkpoint (load Gemma2)
    checkpoint = 'google/gemma-2-2b-it' #model checkpoint of Gemma2

    ## Load tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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


You can generate texts and extract the internal representation of Gemma2 using the following code. You need to specify the directory to save the hidden states and the file name to save the generated texts.

.. note::
    We recommend users to use loop to process each prompt rather than giving the batch of prompts to LLM. This is because LLM may generate responses based on the entire prompts in the batch, which can invalidate the independent assumptions of the generated texts.

.. code-block:: python

    # define the system prompt
    # the system prompt is a text that instructs the LLM to generate texts
    instruction = "You are a text generator who always produces the texts suggested by the prompts."

    # the generated texts are saved in the list
    generated_texts = []

    for k, prompt in enumerate(prompts):
        ######### STEP 1: Generate texts #########
        ## define the input messages
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]

        # tokenize the messages
        # to(model.device): load the tokenized messages onto the device (GPU or CPU) where the model is located
        # this is necessary to ensure that the model can process the input data
        input = tokenizer.apply_chat_template(
            messages,
            # tokenizers option
            add_generation_prompt=False,
            return_dict = True,
            return_tensors = "pt",
        ).to(model.device)
        input_ids = input['input_ids'].to(model.device)
        attention_mask = input['attention_mask'].to(model.device)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            # generation options
            max_new_tokens=512, # maximum number of tokens to generate

            # For deterministic decoding
            do_sample=False,
            top_p = None,
            temperature = None,

            # Padding Token (depends on the model)
            pad_token_id=tokenizer.eos_token_id,

            # For extracting the internal representation
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Save Texts
        response = outputs.sequences[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True)
        generated_texts.append(text)

        ######### STEP 2: Extract Hidden States #########
        hidden_all = outputs.hidden_states[-1][-1].flatten()
        torch.save(hidden_all, f"{save_hidden}/{prefix_hidden}{k}.pt")

In the previous code, we save the internal representations of the last layer corresponding to the **last token**. You can also save the hidden states of other layers by changing the index of ``outputs.hidden_states``. For example, if you want to save the hidden states of the first layer, you can use ``outputs.hidden_states[0][-1]``. You can also save the mean of al the hidden states in the last layer by the following code.

.. code-block:: python

    hidden_all = torch.stack([item[-1] for item in outputs.hidden_states[1:]]).view(-1, 4096)
    hidden_all = hidden_all.mean(dim=0).view(-1, 4096)