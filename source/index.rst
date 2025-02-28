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
   </p>

.. image:: /_static/images/gpi.png
   :alt: logo
   :width: 200px
   :align: center

**gpi_pack** is a Python library for the statistical inference powered by Generative Artificial Intelligence (AI). It provides a set of tools and utilities for performing statistical inference using the internal representation of the Generative AI models. The library is designed to be easy to use and flexible, allowing users to perform a wide range of statistical analyses.

.. note::

   We released **gpi_pack** version 0.1.0 on February 27th, 2025. This is the first version of the package, and we currently only support the setting of Text-as-Treatment based on `our paper <https://arxiv.org/abs/2410.00903>`_. We have been working hard to make this package as useful and user-friendly as possible. If you have any feedback or suggestions, please feel free to reach out to `the maintainer <https://k-nakam.github.io/>`_.

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

How to cite
---------
- Imai, Kosuke and Nakamura, Kentaro (2024). Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments. arXiv preprint arXiv:2410.00903. `[Paper] <https://arxiv.org/abs/2410.00903>`_

.. code-block:: bibtex

   @article{imai2024causal,
     title={Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments},
     author={Imai, Kosuke and Nakamura, Kentaro},
     journal={arXiv preprint arXiv:2410.00903},
     year={2024}
   }