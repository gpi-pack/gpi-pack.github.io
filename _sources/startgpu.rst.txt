.. _gpu_usage_section:

How to use GPU
===============

**gpi_pack** is built upon PyTorch, which is a popular deep learning framework that supports GPU acceleration. The use of GPU is not essential for the statistical inference, but to use LLM to extract the hidden states, you must need to use GPU. The GPU can significantly speed up the computation of the model, especially for large models and datasets.

What's GPU?
----------
GPU (Graphics Processing Unit) is a specialized hardware designed to accelerate the computation of deep learning models. You need to install the GPU driver and CUDA toolkit to use GPU.

The easy way to check the availability of GPU is to run the following command in Python:

.. code-block:: python

    import torch
    print(torch.cuda.is_available())

If the output is ``True``, you can use GPU. If the output is ``False`` but your machine has GPU, you need to install the GPU driver and CUDA toolkit. You can find the instructions for installing the GPU driver and CUDA toolkit on `the NVIDIA website <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/>`_.

What if you do not have GPU?
----------
In many cases, your laptop or desktop does not have GPU. In that case, you can use the cloud service such as `Google Colaboratory <https://colab.research.google.com/>`_ or `Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_. These services provide a virtual environment with GPU support, and you can use them to run your code. Below I show how to use Google Colaboratory.

Google Colaboratory
-------------------
Google Colaboratory is a free cloud service that provides a virtual environment with GPU support. You can use it to run your code without installing anything on your local machine. To use Google Colaboratory, you need to have a Google account.

1. Go to `Google Colaboratory <https://colab.research.google.com/>`_ and sign in with your Google account.

2. Create a new notebook by clicking on the "New Notebook" button. In the notebook, you can write and run Python code just like you would on your local machine.

3. To use GPU, you need to enable it by clicking on the "Runtime" button and selecting "Change runtime type". Then, select "GPU (e.g., T4GPU, A100 GPU)" You can now run your code on the GPU.

.. image:: /_static/images/google_cloud.gif
   :alt: Screenshot of Google Colaboratory
   :width: 600px

4. To install **gpi_pack** on Google Colaboratory, you can use the following command (you can run this directly on the notebook):

.. code-block:: bash

    !pip install gpi_pack

5. To connect your Google Colaboratory notebook to your Google Drive, you need to run the following command:

.. code-block:: python

    from google.colab import drive
    drive.mount('/content/drive')