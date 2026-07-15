.. _ref_TextAsConfounder:

Text-As-Confounder
==================

Text-As-Confounder is another key setting where **gpi_pack** is useful. Text can contain rich pretreatment information, but directly modeling all textual features is difficult because the data are high-dimensional and can lead to poor overlap (also known as positivity). By combining LLM internal representations with the learned **deconfounder**, we can adjust for the outcome-relevant information in the text without directly specifying every confounding feature.

.. note::
    This part is based on our paper `GenAI-Powered Inference <https://arxiv.org/abs/2507.03897>`_. Please refer to the paper for the technical details.

What is Text-As-Confounder?
---------------------------
Text-As-Confounder refers to settings in which pretreatment text contains information useful for causal adjustment. If every textual detail is controlled directly, observations can become nearly unique and comparable treated and control units may be difficult to find. The goal is therefore to retain the text features that are useful confounders while reducing irrelevant high-dimensional variation.

To address this challenge, `our paper <https://arxiv.org/abs/2507.03897>`_ demonstrates that the same estimator used for Text-as-Treatment can be applied to Text-as-Confounder. ``TarNet`` learns a shared representation from the LLM features, appends a supplied treatment value, and predicts the corresponding outcome with one treatment-conditioned outcome network. The estimator evaluates that network under treatment zero and one to obtain the two potential-outcome predictions.

Once the deconfounder and treatment-conditioned outcome model are estimated, a propensity score model is built based on the deconfounder. Finally, the estimated nuisance functions are combined with a doubly robust score to estimate the treatment effect.

How to estimate treatment effects
---------------------------------

Use the same ``estimate_k_ate`` wrapper as in the Text-as-Treatment setting. In this application, ``R`` contains representations of pretreatment text, while ``T`` is a separate binary intervention. Please refer to :ref:`ref_TextAsTreatment` for the workflow and :ref:`ref_estimate_k_ate` for the complete function reference.
