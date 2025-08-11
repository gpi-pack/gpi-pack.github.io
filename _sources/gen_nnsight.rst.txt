Generating Texts without GPU
===========


In many cases, you do not have access to a GPU, but you still want to use GPI. In such cases, `nnsight <https://nnsight.net/>`_ offers a solution. nnsight is a Python package that allows you to generate texts and extract the internal representations of various open-source LLMs using API. This package is particularly useful for users who want to perform GPI on machines without GPU support.

.. note::
    nnsight is a third-party package, and it requires an API key to use. See `nnsight documentation <https://nnsight.net/start/>`_ for more details on how to obtain an API key and use the package.

Below, we will show how to use nnsight to generate texts and extract the internal representations of LLMs using `LLaMa3.3-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_.

NNsight for GPI
---------

Firstly, you need to install the nnsight package. You can install it using pip:

.. code-block:: bash

    pip install nnsight

You also need to obtain an API key from nnsight. You can get the API key by signing up on the nnsight website. Once you have the API key, you can set it in your code as follows:

.. code-block:: python

    # log in to nnsight
    from nnsight import CONFIG
    CONFIG.API.APIKEY = input("Enter your API key: ") # Enter your API key here

Once you log in to nnsight, you can use it to load the models. Below, we load the LLaMa3.3-70B-Instruct model and its tokenizer. Note that for LLaMa model, you need to log in to Huggingface to access the model.

.. code-block:: python

    # loading required packages
    from nnsight import LanguageModel

    ## Specify checkpoint (load LLaMa3.3-70B-Instruct)
    checkpoint = 'meta-llama/Llama-3.3-70B-Instruct' # model checkpoint of LLaMa3.3-70B-Instruct

    ## Load tokenizer and pretrained model
    model = LanguageModel("meta-llama/Llama-3.3-70B-Instruct", device_map="auto")
    tokenizer = model.tokenizer

You can do this by running the following command in your terminal:

.. code-block:: bash

    huggingface-cli login

Now, you are ready to generate texts and extract the internal representations of LLaMa3.3-70B-Instruct using nnsight.
Suppose that you have the following list of prompts.

.. code-block:: python

    prompts = [
        'Create a biography of an American politician named Nathaniel C. Gilchrist',
        'Create a biography of an American politician named John Doe',
        'Create a biography of an American politician named Jane Smith',
        'Create a biography of an American politician named Mary Johnson',
        'Create a biography of an American politician named Robert Brown',
    ]


You can generate texts and extract the internal representation using the following code. You need to specify the directory to save the hidden states and the file name to save the generated texts.

.. note::
    We recommend users to use loop to process each prompt rather than giving the batch of prompts to LLM. This is because LLM may generate responses based on all the prompts in the batch, which can invalidate the independent assumptions of the generated texts.

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
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_dict = True,
            return_tensors = "pt",
        )

        with model.generate(input_ids, max_new_tokens=4096, do_sample = False, remote=True) as tracer:
            hidden_states = model.model.layers[-1].output.save() #-1 for last layer
            out = model.generator.output.save()

        # Save Texts
        text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
        generated_texts.append(text)

        ######### STEP 2: Extract Hidden States #########
        hidden_all = hidden_states[0][0][-1].flatten()
        torch.save(hidden_all, f"{save_hidden}/{prefix_hidden}{k}.pt")

