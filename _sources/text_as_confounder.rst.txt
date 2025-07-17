.. _ref_TextAsConfounder:

Text-As-Confounder
===========

Text-As-Confounder is another key setting where **gpi_pack** is useful. As more and more text data are readily available, it is often possible that we can control many confounders by controlling the information contained in the text data. However, it is often the case that the direct modeling of text data is not feasible due to the high-dimensionality of the text data and is also problematic for causal inference perspective as it typically violates so-called overlap condition (also known as positivity). By combining the use of internal representations of LLMs and the representation learning method called **deconfounder**, we can estimate the treatment effects of the feature of interest without directly observing all confounding features.

.. note::
    This part is based on our paper `GenAI-Powered Inference <https://arxiv.org/abs/2507.03897>`_. Please refer to the paper for the technical details.

What is Text-As-Confounder?
---------
Text-As-Confounder refers to scenarios where the text data contains many features that is useful to control for confounding effects in causal inference. A primary challenge in this setting is that text data contains a large number of features, and thus if we directly control all features of texts, we can easily violate the overlap condition, which is a key assumption in causal inference. For example, if we want to control how people write about the government in their social media posts, if we directly control all the social media posts, as everyone writes differently, it is conceptually hard to find the counterpart of the social media posts that writes the exact same way but in a different treatment condition. Thus, in many cases, we want to control only the part of texts that works as a confounder.

To address this challenge, `our paper <https://arxiv.org/abs/2507.03897>`_ demonstrates that the same methods for Text-as-Treatment example can be applied to Text-as-Confounder. Specifically, we propose to use LLMs’ internal representations to bypass the direct modeling of texts and learn only the representations of confounding features called  **deconfounder**, which enables us to estimate the treatment effects of the feature of interest without directly observing all confounding features. As before, the deconfounder is estimated using `TarNet`, which takes the internal representations as input and predicts outcomes under both treatment and control conditions using a shared deconfounder.

Once the deconfounder and the outcome models are estimated, a propensity score model is built based on the deconfounder. Finally, the estimated outcome models and propensity score model are used together with double machine learning techniques to estimate treatment effects.

How to estimate treatment effects
---------

We can use the same wrapper function in **gpi_pack** as in the Text-as-Treatment setting to estimate treatment effects. Please refer to :ref:`ref_TextAsTreatment` for the details of the function.