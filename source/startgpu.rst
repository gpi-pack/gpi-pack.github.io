.. _gpu_usage_section:

How to Use a GPU
================

**gpi_pack** uses PyTorch for its neural models. A GPU is not required: the
statistical estimators, LLM workflow, Stable Diffusion extractor, and Cosmos
extractor can all select CPU execution. In practice, generating
representations with a large language, image, or video model is usually much
faster on a compatible accelerator and may require more memory than a laptop
provides.

The device rules depend on the interface:

- ``TarNet`` and the image and video extractors select CUDA when it is
  available and otherwise use the CPU.
- Dynamic GPI with ``device="auto"`` tries CUDA, then Apple MPS, then CPU.
- A Transformers language model uses the device selected when you load it.

Checking CUDA
-------------

An NVIDIA GPU uses CUDA. Check whether the installed PyTorch build can access
one with:

.. code-block:: python

   import torch

   print(torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))

If this prints ``False`` on a machine with an NVIDIA GPU, install a PyTorch
build compatible with the machine's driver by following the `PyTorch
installation selector <https://pytorch.org/get-started/locally/>`_. Installing
the CUDA toolkit alone does not change a CPU-only PyTorch installation.

Using a Hosted GPU
------------------

If you do not have a local GPU, you can use a hosted notebook such as `Google
Colab <https://colab.research.google.com/>`_ or a cloud service such as Amazon
SageMaker. Colab offers optional GPU runtimes, but the available hardware and
usage limits can change.

To use a Colab GPU:

1. Create a notebook and select **Runtime > Change runtime type**.

2. Select a GPU hardware accelerator and reconnect the runtime.

3. Run ``torch.cuda.is_available()`` in the notebook to verify that PyTorch
   can see the device.

4. Install **gpi_pack** from PyPI. For example, include both video and tuning
   dependencies with:

   .. code-block:: bash

      !python -m pip install --upgrade "gpi-pack[video,tune]"

.. image:: /_static/images/google_cloud.gif
   :alt: Selecting a GPU runtime in Google Colab
   :width: 600px

If your data are in Google Drive, mount it separately:

.. code-block:: python

   from google.colab import drive

   drive.mount("/content/drive")

Using Remote Model Execution
----------------------------

A local GPU is not the only option for LLM representations. The :doc:`NNsight
guide <gen_nnsight>` explains how to run a supported open-weight model on the
remote NDIF service and download only the values explicitly saved by the
trace.
