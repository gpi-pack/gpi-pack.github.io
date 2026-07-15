.. _ref_CosmosVideoExtractor:

CosmosVideoExtractor
====================

Description
-----------

The ``CosmosVideoExtractor`` class loads a continuous NVIDIA Cosmos VAE and provides methods for encoding, reconstructing, and extracting representations from RGB video clips. Reusing one instance avoids loading the model for every video.

Parameters
----------

- ``model_id`` (*str*, optional): Hugging Face checkpoint. The default is ``"nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"``.
- ``device`` (*str* or *torch.device*, optional): execution device. If ``None``, CUDA is used when available and CPU otherwise.
- ``cache_dir`` (*str* or *pathlib.Path*, optional): model cache directory.
- ``dtype`` (*str* or *torch.dtype*, optional): ``"auto"``, ``"bf16"``, ``"fp16"``, ``"fp32"``, or a PyTorch dtype. The default is ``"auto"``.
- ``frame_size`` (*tuple of int*, optional): input ``(height, width)``. No resizing is applied when ``None``.
- ``max_frames`` (*int*, optional): maximum frames retained by uniform sampling. The default is 121; ``None`` disables sampling.
- ``pad_multiple`` (*int*, optional): spatial padding multiple. If ``None``, the VAE's spatial compression ratio is used.
- ``temporal_pooling`` (*str*, optional): ``"temporal_mean"`` or ``"none"``. The default is ``"temporal_mean"``.
- ``vae`` (*object*, optional): compatible injected VAE, mainly for custom models and testing.
- ``pretrained_kwargs`` (*mapping*, optional): extra Hugging Face loader options, such as ``token`` or ``revision``.

Example Usage
-------------

.. code-block:: python

   import numpy as np
   from gpi_pack.video import CosmosVideoExtractor

   frames = np.zeros((9, 320, 480, 3), dtype=np.uint8)
   extractor = CosmosVideoExtractor(frame_size=(320, 480))

   result = extractor.reconstruct_video(frames)
   print(result.representation.shape)
   extractor.save_video(result.reconstruction, "reconstructed.mp4", fps=30)

Methods
-------

preprocess_video
^^^^^^^^^^^^^^^^

``preprocess_video(frames)`` converts RGB ``[T, H, W, 3]`` frames, or one ``[H, W, 3]`` image, into a normalized ``[1, 3, T, H, W]`` tensor after optional sampling and resizing.

encode_video
^^^^^^^^^^^^

``encode_video(frames)`` encodes a clip without running the decoder and returns a :ref:`ref_VideoExtractionResult`. Its ``reconstruction`` field is ``None``.

decode_latents
^^^^^^^^^^^^^^

``decode_latents(latent, pad_bottom=0, pad_right=0, num_frames=None)`` decodes a latent tensor, removes preprocessing padding, optionally truncates the time dimension, and returns a BCTHW tensor.

reconstruct_video
^^^^^^^^^^^^^^^^^

``reconstruct_video(frames)`` encodes and decodes a clip and returns its latent, decoder input, pooled representation, reconstruction, and preprocessing metadata.

transform_video
^^^^^^^^^^^^^^^

``transform_video(frames)`` is an alias for ``reconstruct_video`` and matches the image extractor's naming convention.

save_video
^^^^^^^^^^

``save_video(video, path, fps)`` saves one reconstructed ``[1, 3, T, H, W]`` tensor as a visual-only H.264 MP4.

process_video
^^^^^^^^^^^^^

``process_video(video_path, output_hidden_dir, output_video_dir=None, segment_seconds=..., segment=None, drop_last=False, save_latent=False, save_decoder_input=False, overwrite=False, verbose=True)`` segments one file, writes each representation payload and optional reconstruction, and returns a list of :ref:`ref_VideoSegmentOutput` objects.

.. note::
   The model uses the deterministic latent distribution mode. Audio is not processed or written to reconstructed files.
