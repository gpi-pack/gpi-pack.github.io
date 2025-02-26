GPI: Generative-AI Powered Inference
============================================

.. raw:: html

   <p>
   <a href="https://pypi.org/project/gpi_pack/">
      <img src="https://img.shields.io/pypi/v/gpi_pack.svg" alt="PyPI version">
   </a>
   <a href="https://pypi.org/project/gpi_pack/">
      <img src="https://img.shields.io/pypi/pyversions/gpi_pack.svg" alt="Python Versions">
   </a>
   <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
   </a>
   <a href="https://github.com/k-nakam/gpi_pack">
      <img src="https://img.shields.io/github/stars/k-nakam/gpi_pack?style=social" alt="GitHub stars">
   </a>
   </p>

**gpi_pack** is a Python library for the statistical inference powered by Generative Artificial Intelligence (AI). It provides a set of tools and utilities for performing statistical inference using the internal representation of the Generative AI models. The library is designed to be easy to use and flexible, allowing users to perform a wide range of statistical analyses.

.. note::

   We released **gpi_pack** version 0.1.0 on 2025-03-01. This is the first version of the package, and we currently only support the setting of Text-as-Treatment based on `our paper <https://arxiv.org/abs/2410.00903>`_. We have been working hard to make this package as useful and user-friendly as possible. If you have any feedback or suggestions, please feel free to reach out to `the maintainer <https://k-nakam.github.io/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   gpi
   startgpu

.. toctree::
   :maxdepth: 2
   :caption: Data Generation

   gen_llama
   gen_llm

.. toctree::
   :maxdepth: 2
   :caption: Basic Operation

   tarnet


.. toctree::
   :maxdepth: 2
   :caption: Advanced Operations

   hyperparameter
   custom_imp
   quantization
   reference

.. toctree::
   :maxdepth: 2
   :caption: References

   f_dml_score
   f_estimate_k_ate
   f_estimate_psi_split
   f_extract_and_save_hiddens
   f_generate_text
   f_get_instructions
   f_load_hiddens
   f_save_generated_texts
   f_SpectralNormClassifier
   f_TarNet_loss
   f_TarNet
   f_TarNetBase
   f_TarNetHyperparameterTuner
