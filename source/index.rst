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

   We released **gpi_pack** version 0.2.0 on July 14th. We currently support (1) Text/Image-as-Confounder, (2) Text/Image-as-Treatment, and (3) Video-as-Treatment with scalar or repeated outcomes. Please read `our recent paper <https://arxiv.org/abs/2607.06126>`_ for the application examples. We have been working hard to make this package as useful and user-friendly as possible. If you have any feedback or suggestions, please feel free to reach out to `the maintainer <https://k-nakam.github.io/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   gpi
   startgpu
   report

.. toctree::
   :maxdepth: 2
   :caption: Data Generation

   gen_llama
   gen_llm
   gen_nnsight
   gen_diffusion
   gen_cosmos
   gen_whisper

.. toctree::
   :maxdepth: 2
   :caption: Basic Operations

   tarnet
   text_as_confounder
   image_as_treatment
   video_as_treatment


.. toctree::
   :maxdepth: 2
   :caption: Advanced Operations

   hyperparameter
   custom_imp
   quantization

.. toctree::
   :maxdepth: 2
   :caption: References

   reference
   f_dml_score
   f_estimate_k_ate
   f_estimate_psi_split
   f_extract_and_save_hiddens
   f_generate_text
   f_get_instructions
   f_load_hiddens
   f_save_generated_texts
   f_pad_to_multiple_of_8
   f_StableDiffusionImg2ImgExtractor
   f_extract_images
   f_CosmosVideoExtractor
   f_VideoExtractionResult
   f_VideoSegmentOutput
   f_extract_videos
   f_TextMLPEncoder
   f_Video3DEncoder
   f_DynamicTarNetBase
   f_mse_loss
   f_DynamicTarNet
   f_DynamicGPIHyperparameterTuner
   f_estimate_k_ipsi
   f_SpectralNormClassifier
   f_TarNet
   f_TarNetBase
   f_TarNetHyperparameterTuner

How to cite
-----------
- Imai, Kosuke and Nakamura, Kentaro (2024). Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments. arXiv preprint arXiv:2410.00903. `[Paper] <https://arxiv.org/abs/2410.00903>`_

.. code-block:: bibtex

   @article{imai2024causal,
     title={Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments},
     author={Imai, Kosuke and Nakamura, Kentaro},
     journal={arXiv preprint arXiv:2410.00903},
     year={2024}
   }

- Imai, Kosuke and Nakamura, Kentaro (2025). GenAI-Powered Inference. arXiv preprint arXiv:2507.03897. `[Paper] <https://arxiv.org/abs/2507.03897>`_

.. code-block:: bibtex

   @article{imai2025genai,
      title={GenAI-Powered Inference},
      author={Imai, Kosuke and Nakamura, Kentaro},
      journal={arXiv preprint arXiv:2507.03897},
      year={2025}
   }

- Nakamura, Kentaro and Imai, Kosuke (2026). GenAI Powered Dynamic Causal Inference with Unstructured Data. arXiv preprint arXiv:2605.07834. `[Paper] <https://arxiv.org/abs/2605.07834>`_

.. code-block:: bibtex

   @article{nakamura2026genai,
     title={GenAI Powered Dynamic Causal Inference with Unstructured Data},
     author={Nakamura, Kentaro and Imai, Kosuke},
     journal={arXiv preprint arXiv:2605.07834},
     year={2026}
   }

- Nakamura, Kentaro, Breuer, Adam, Crespin, Michael H., Dietrich, Bryce J., and Imai, Kosuke (2026). Causal Inference with Video Features as Treatments. arXiv preprint arXiv:2607.06126. `[Paper] <https://arxiv.org/abs/2607.06126>`_

.. code-block:: bibtex

   @article{nakamura2026causal,
     title={Causal Inference with Video Features as Treatments},
     author={Nakamura, Kentaro and Breuer, Adam and Crespin, Michael H. and Dietrich, Bryce J. and Imai, Kosuke},
     journal={arXiv preprint arXiv:2607.06126},
     year={2026}
   }
