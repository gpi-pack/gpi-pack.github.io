What's GPI?
===============

**gpi_pack** is a Python package for Generative-AI Powered Inference. This section provides an overview of the package, its purpose, and the workflow.

Generative-AI Powered Inference
---------
Generative-AI Powered Inference (GPI) is a method for causal inference that leverages the internal representations of the deep generative models for the statistical inference. Traditionally, the statistical inference of unstructured objects like texts or images require researchers to model texts or images directly. However, given the high-dimensional nature of the objects, it is often difficult to model them directly. GPI circumvents this problem by using **the internal representations** of the deep generative models. Because we know that the internal representation actually generate the texts, these internal representation gives you the low-dimensional representation of the unstructured objects without estimation errors.

The entire GPI procedure consists of the following two steps:

1. Generate the internal representation of the unstructured objects using the deep generative models.
2. Statistical inference using the internal representation.

Sometimes, we need the additional step for the statistical inference, because the unstructured objects like texts or images contain so many features in a complex manner. Thus, we need to extract the variable of interest from the internal representaiton, which is called **Representation Learning**. For example, for Text-As-Treatment where we randomly assign many texts to the survey participants and try to infer the causal effect of one specific feature of interest, other features in the texts might act as a confounder. However, if we directly control the entire internal representation, we will violate the overlap violation. As a result, it is essential to learn the representation of the confounding features so that we do not violate the positivity.

This package provides the implementation of the GPI procedure, including the generation of the internal representation, the statistical inference, and the representation learning. The package is designed to be user-friendly and accessible to researchers and practitioners across disciplines.

Contribute to GPI
---------
We welcome your feedback, comments, suggestions, code, and bag reports. You can contibute to GPI by

- Issue bag reports and wish lists on `Github Issues <https://github.com/k-nakam/gpi-pack/issues>`_
- Email to `the maintainer <mailto:knakamura@g.harvard.edu>`_.

References
---------
- Imai, Kosuke and Nakamura, Kentaro (2024). Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments. arXiv preprint arXiv:2410.00903. `[Paper] <https://arxiv.org/abs/2410.00903>`_