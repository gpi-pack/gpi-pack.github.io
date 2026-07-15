Function Reference
==================

This section explains the package-owned public functions and classes in **gpi_pack**. The entries are grouped by module so that you can find the lower-level function used by each workflow. Underscore-prefixed modules and names are implementation details and are not part of the user-facing API.

Text Generation
---------------

- :doc:`f_get_instructions`: creates the built-in system instruction for creating or repeating text, or returns a custom instruction.
- :doc:`f_generate_text`: generates text with non-sampling decoding and saves a pooled LLM hidden state for every prompt.
- :doc:`f_save_generated_texts`: saves generated texts and their prompts in a pickle file.
- :doc:`f_extract_and_save_hiddens`: runs the complete text-generation, hidden-state extraction, and output-saving workflow.

Image Generation
----------------

- :doc:`f_pad_to_multiple_of_8`: pads a PIL image so that both spatial dimensions are divisible by eight.
- :doc:`f_StableDiffusionImg2ImgExtractor`: loads the Stable Diffusion components and provides preprocessing, encoding, transformation, decoding, and saving methods.
- :doc:`f_extract_images`: processes one or more image-prompt pairs and saves the final diffusion latents and optional generated images.

Video Processing
----------------

- :doc:`f_CosmosVideoExtractor`: loads the Cosmos VAE and provides in-memory encoding, reconstruction, representation extraction, and file-processing methods.
- :doc:`f_VideoExtractionResult`: stores the tensors and preprocessing metadata returned for an in-memory video clip.
- :doc:`f_VideoSegmentOutput`: stores the output paths and segment identity returned by file processing.
- :doc:`f_extract_videos`: discovers, segments, and processes video files and saves one Cosmos representation payload for every selected segment.

Static Inference
----------------

- :doc:`f_TarNetBase`: implements the treatment-conditioned neural outcome model and learned deconfounder.
- :doc:`f_TarNet`: trains, validates, and predicts with ``TarNetBase``.
- :doc:`f_SpectralNormClassifier`: estimates class probabilities with a spectrally normalized neural network.
- :doc:`f_dml_score`: calculates the doubly robust influence score for an average treatment effect.
- :doc:`f_estimate_psi_split`: cross-fits the propensity model within a sample and returns influence scores and propensity predictions.
- :doc:`f_estimate_k_ate`: estimates an average treatment effect and its standard error with k-fold cross-fitting.
- :doc:`f_TarNetHyperparameterTuner`: tunes the static TarNet outcome model with Optuna and can refit the best configuration.
- :doc:`f_load_hiddens`: loads saved ``.pt`` representations in a requested order and returns a NumPy array.

Dynamic Inference
-----------------

- :doc:`f_TextMLPEncoder`: maps each vector or text representation to a fixed-width segment embedding.
- :doc:`f_Video3DEncoder`: maps each video latent volume to a fixed-width segment embedding with 3D convolutions.
- :doc:`f_DynamicTarNetBase`: implements the masked sequential representation and outcome networks used by Dynamic GPI.
- :doc:`f_mse_loss`: calculates the scalar mean squared error for two compatible tensors.
- :doc:`f_DynamicTarNet`: trains, validates, and predicts one scalar outcome with the dynamic sequence model in vector-only or multimodal mode.
- :doc:`f_DynamicGPIHyperparameterTuner`: tunes and refits the scalar-outcome ``DynamicTarNet`` with Optuna. ``DynamicTarNetHyperparameterTuner`` is an alias of the same class.
- :doc:`f_estimate_k_ipsi`: estimates cross-fitted longitudinal incremental-intervention curves and uncertainty for scalar or repeated outcomes.
