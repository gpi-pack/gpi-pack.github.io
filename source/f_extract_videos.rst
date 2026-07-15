.. _ref_extract_videos:

extract_videos
==============

Description
-----------

The ``extract_videos`` function is the high-level Cosmos pipeline. It discovers video files, divides each file into segments, extracts a deterministic Cosmos representation for every selected segment, and saves a metadata-rich ``.pt`` payload. Supplying an output video directory also saves visual-only reconstructions.

Arguments
---------

- ``videos`` (*path or sequence of paths*): video file, recursively searched directory, or a sequence containing either type (**required**).
- ``output_hidden_dir`` (*path*): root directory for representation payloads (**required**).
- ``segment_seconds`` (*float*): positive segment duration in seconds (**required**, keyword-only).
- ``output_video_dir`` (*path*, optional): reconstruction directory. ``None`` skips decoding.
- ``segment`` (*int* or *tuple*, optional): zero-based segment or inclusive ``(start, end)`` range. The end can be ``None``.
- ``model_id`` (*str*, optional): Cosmos checkpoint. The default is ``"nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"``.
- ``device`` (*str* or *torch.device*, optional): execution device.
- ``cache_dir`` (*path*, optional): model cache directory.
- ``dtype`` (*str* or *torch.dtype*, optional): model precision. The default is ``"auto"``.
- ``frame_size`` (*tuple of int*, optional): common ``(height, width)``. The default preserves each source size.
- ``max_frames`` (*int* or *None*, optional): maximum frames per segment. The default is 121.
- ``pad_multiple`` (*int*, optional): spatial model-input multiple. The default is read from the VAE.
- ``temporal_pooling`` (*str*, optional): ``"temporal_mean"`` saves ``[C, H, W]``; ``"none"`` saves ``[C, D, H, W]``.
- ``drop_last`` (*bool*, optional): whether to drop a short final segment. The default is ``False``.
- ``save_latent`` (*bool*, optional): whether to include the encoder latent. The default is ``False``.
- ``save_decoder_input`` (*bool*, optional): whether to include the unpooled decoder input. The default is ``False``.
- ``overwrite`` (*bool*, optional): whether to replace existing files. The default is ``False``.
- ``extractor`` (:ref:`ref_CosmosVideoExtractor`, optional): existing model instance to reuse.
- ``pretrained_kwargs`` (*mapping*, optional): extra model-loading options.
- ``verbose`` (*bool*, optional): whether to print progress. The default is ``True``.

Returns
-------

- *list of VideoSegmentOutput*: ordered output records containing the source path, segment index, representation path, and optional reconstruction path.

Example Usage
-------------

.. code-block:: python

   import torch
   from gpi_pack.video import extract_videos

   outputs = extract_videos(
       videos="path/to/videos",
       output_hidden_dir="outputs/hidden",
       segment_seconds=5,
       frame_size=(320, 480),
   )

   payload = torch.load(
       outputs[0].representation_path,
       map_location="cpu",
       weights_only=True,
   )
   representation = payload["representation"]

Related Module Utilities
------------------------

The following functions support the high-level pipeline. They remain available from ``gpi_pack.video`` for specialized workflows, although they are not re-exported as the module's primary public API:

- ``parse_segment_spec`` and ``format_segment_spec`` parse and display CLI segment ranges.
- ``model_dtype`` validates and resolves a precision option for a device.
- ``find_videos`` recursively discovers supported video files.
- ``fps_for`` selects a finite nominal frame rate from a PyAV stream.
- ``segment_meta`` creates the frame and time metadata for one segment.
- ``iter_segments`` yields consecutive RGB frame arrays and their metadata.
- ``uniform_frame_indices`` selects at most a requested number of frames while retaining both endpoints.
- ``frames_to_bcthw`` validates and normalizes RGB frames into a BCTHW tensor.
- ``resize_spatial``, ``pad_spatial``, and ``crop_spatial`` implement spatial preprocessing and restoration.
- ``get_latent`` returns the deterministic Cosmos latent distribution mode.
- ``pool_decoder_input`` applies temporal-mean pooling or preserves latent time.
- ``reconstruct_and_extract`` returns the latent, decoder input, and reconstruction from a compatible VAE.
- ``bcthw_to_uint8`` converts one normalized BCTHW reconstruction to uint8 THWC frames.
- ``save_mp4`` writes visual-only H.264 video frames.
- ``output_name`` creates a readable collision-resistant directory name for a source video.
- ``parse_args`` and ``main`` implement the ``gpi-extract-video`` command-line interface.
