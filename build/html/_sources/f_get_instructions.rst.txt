.. _ref_get_instructions:

get_instruction
===============

Purpose and Description
----------------------------
The ``get_instruction`` function generates an instruction string based on a given task type. This string guides the language model’s behavior during text generation. For example, if the task type is ``"create"``, the instruction tells the model to generate text; if ``"repeat"``, it instructs the model to simply repeat the provided input; any other string is treated as a custom task instruction.

Arguments
---------
- **task_type** (*str*): The type of task to be performed. Expected values include:

  - ``"create"`` – instructs the model to create texts.
  - ``"repeat"`` – instructs the model to repeat the input text.
  - Any other string is interpreted as a user-specific task.

Returns
-------
- **instruction** (*str*): The instruction string corresponding to the provided task type.

Example Usage
-------
.. code-block:: python

    from gpi_pack.llm import get_instruction

    # Example for a creation task:
    instruction = get_instruction("create")
    print(instruction)
    # Output: "You are a text generator who always produces the texts suggested by the prompts."

