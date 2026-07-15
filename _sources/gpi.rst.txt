What Is GPI?
============

**gpi_pack** implements GenAI-Powered Inference (GPI), a framework for
statistical inference with unstructured data such as text, images, and video.
Instead of modeling the raw object directly, GPI uses an internal
representation from a deep generative model as a structured input to the
statistical analysis.

GPI Workflow
------------

The workflow has two main stages:

1. **Representation generation.** Generate or reconstruct each unstructured
   object with a fixed generative model and extract the package-specific
   internal representation. The text workflow saves an LLM generation state,
   the image workflow saves the final Stable Diffusion latent, and the video
   workflow saves the Cosmos decoder input with optional temporal pooling.

2. **Statistical inference.** Fit the relevant outcome and nuisance models to
   estimate the target effect and its uncertainty. The package provides a
   static estimator for one binary treatment per unit and a dynamic estimator
   for sequences of binary treatments with scalar or repeated outcomes.

Depending on the estimand, an additional **representation-learning** step is
used inside the outcome model. An unstructured object can contain the treatment
feature of interest together with other features related to treatment and
outcome. The TarNet models learn a lower-dimensional representation used to
adjust for those features while fitting treatment-conditioned outcomes.

Current Package Scope
---------------------

Version 0.2.1 provides:

- local LLM text generation and representation extraction for models with a
  4096-wide hidden state, plus manual examples for other local or remote
  models;
- Stable Diffusion 1.x/2.x image reconstruction and latent extraction;
- NVIDIA Cosmos video segmentation, deterministic encoding, reconstruction,
  and representation extraction;
- static Text/Image-as-Treatment and Text/Image-as-Confounder inference;
- dynamic Video-as-Treatment inference with scalar or repeated outcomes; and
- Optuna-based tuning helpers for the static and scalar-outcome dynamic neural
  models.

The representation is fixed by the checkpoint, model revision, preprocessing,
precision, generation settings, pooling rule, and software environment. Keep
these choices consistent across observations and record them for
reproducibility. The methodological papers listed on the home page give the
identification assumptions and statistical guarantees; installing the package
does not make those assumptions hold automatically for a particular study.
