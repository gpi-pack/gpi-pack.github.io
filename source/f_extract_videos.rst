.. _ref_extract_videos:

extract_videos
==============

Description
-----------

The ``extract_videos`` function is the high-level Cosmos pipeline. It discovers video files, divides each file into segments, extracts a deterministic Cosmos representation for every selected segment, and saves a metadata-rich ``.pt`` payload. Supplying an output video directory also saves visual-only reconstructions.

.. code-block:: text

   extract_videos(
       videos,
       output_hidden_dir,
       *,
       segment_seconds,
       output_video_dir=None,
       segment=None,
       model_id="nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
       device=None,
       cache_dir=None,
       dtype="auto",
       frame_size=None,
       max_frames=121,
       pad_multiple=None,
       temporal_pooling="temporal_mean",
       drop_last=False,
       save_latent=False,
       save_decoder_input=False,
       overwrite=False,
       extractor=None,
       pretrained_kwargs=None,
       verbose=True,
   )

Arguments
---------

- ``videos`` (*path or sequence of paths*): video file, recursively searched directory, or a sequence containing either type (**required**). Supported extensions are ``.avi``, ``.m4v``, ``.mkv``, ``.mov``, ``.mp4``, and ``.webm``, matched case-insensitively.
- ``output_hidden_dir`` (*path*): root directory for representation payloads (**required**).
- ``segment_seconds`` (*float*): finite, positive nominal segment duration in seconds (**required**, keyword-only). The implementation rounds ``segment_seconds * fps`` to choose a fixed number of decoded frames per segment.
- ``output_video_dir`` (*path*, optional): reconstruction directory. ``None`` skips decoding.
- ``segment`` (*int* or *tuple*, optional): non-negative zero-based segment index or inclusive ``(start, end)`` range. The end can be ``None`` for an open range.
- ``model_id`` (*str*, optional): Cosmos checkpoint. The default is ``"nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"``.
- ``device`` (*str* or *torch.device*, optional): execution device.
- ``cache_dir`` (*path*, optional): model cache directory.
- ``dtype`` (*str* or *torch.dtype*, optional): model precision. The default is ``"auto"``.
- ``frame_size`` (*tuple of int*, optional): common positive ``(height, width)``. The default preserves each source size.
- ``max_frames`` (*int* or *None*, optional): positive maximum number of frames retained per segment. The default is 121; ``None`` disables sampling in the Python API.
- ``pad_multiple`` (*int*, optional): positive spatial model-input multiple. The default is read from the VAE.
- ``temporal_pooling`` (*str*, optional): ``"temporal_mean"`` saves ``[C, H, W]``; ``"none"`` saves ``[C, D, H, W]``.
- ``drop_last`` (*bool*, optional): whether to drop a final segment containing fewer than ``round(segment_seconds * fps)`` decoded frames. The default is ``False``.
- ``save_latent`` (*bool*, optional): whether to include the encoder latent. The default is ``False``.
- ``save_decoder_input`` (*bool*, optional): whether to include the unpooled decoder input. The default is ``False``.
- ``overwrite`` (*bool*, optional): whether to replace existing files. The default is ``False``.
- ``extractor`` (:ref:`ref_CosmosVideoExtractor`, optional): existing model instance to reuse. When supplied, the model and preprocessing options already stored on that instance take effect; the corresponding constructor arguments to ``extract_videos`` are not reapplied.
- ``pretrained_kwargs`` (*mapping*, optional): extra model-loading options used only when ``extractor`` is ``None``.
- ``verbose`` (*bool*, optional): whether to print progress. The default is ``True``.

Returns
-------

- *list of VideoSegmentOutput*: ordered output records containing the source path, segment index, representation path, and optional reconstruction path.

Source paths are de-duplicated and sorted before processing, and segments remain in ascending index order. An error is raised when no supported videos are found, when an input file has an unsupported extension, or when an input directory is also used as an output directory. Existing representation or reconstruction files raise ``FileExistsError`` unless ``overwrite=True``.

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

Saved Payload
-------------

Each ``segment_XXXXXX.pt`` file contains an unbatched float32 ``representation`` on CPU. Its shape is ``[C, H, W]`` for ``temporal_pooling="temporal_mean"`` and ``[C, D, H, W]`` for ``temporal_pooling="none"``. The payload also records the source path, model identifier and revision information, library and VAE metadata, source segment frame/time bounds, selected frame indices, input and padded shapes, spatial padding, sampling information, and the processed frame rate used for reconstruction.

``save_latent=True`` adds an unbatched ``latent`` tensor. ``save_decoder_input=True`` adds the unbatched ``decoder_input`` tensor and its compatibility alias ``pre_decoder_hidden``. ``audio_reconstructed`` is always ``False``.

Command-Line Interface
----------------------

The same pipeline is exposed as ``gpi-extract-video`` (or ``python -m gpi_pack.video``):

.. code-block:: bash

   gpi-extract-video \
       --input path/to/videos \
       --output_dir outputs \
       --segment_seconds 5 \
       --segment 1-3 \
       --frame_size 320 480 \
       --save_reconstruction

``--input``, ``--output_dir``, and ``--segment_seconds`` are required. ``--segment`` accepts ``N``, ``START-END``, or ``START-`` and uses inclusive bounds. ``--max_frames 0`` disables frame sampling; negative values are rejected. ``--save_reconstruction`` uses ``--output_dir`` for both payloads and reconstructed MP4 files, while the Python function permits separate roots. Other flags correspond to the options above: ``--model_id``, ``--cache_dir``, ``--device``, ``--dtype``, ``--temporal_pooling``, ``--pad_multiple``, ``--max_frames``, ``--frame_size``, ``--drop_last``, ``--save_latent``, ``--save_decoder_input``, and ``--overwrite``.

Related Module Utilities
------------------------

The following functions support the high-level pipeline. They remain available from ``gpi_pack.video`` for specialized workflows, although they are not re-exported as the module's primary public API:

- ``parse_segment_spec`` and ``format_segment_spec`` parse and display CLI segment ranges.
- ``model_dtype`` validates and resolves a precision option for a device.
- ``find_videos`` recursively discovers supported video files.
- ``fps_for`` selects a finite nominal frame rate from a PyAV stream.
- ``segment_meta`` creates the frame and time metadata for one segment.
- ``iter_segments`` yields consecutive RGB frame arrays and their metadata.
- ``uniform_frame_indices`` selects at most a requested number of uniformly spaced frames. It retains both endpoints when at least two frames are selected.
- ``frames_to_bcthw`` validates and normalizes RGB frames into a BCTHW tensor.
- ``resize_spatial``, ``pad_spatial``, and ``crop_spatial`` implement spatial preprocessing and restoration.
- ``get_latent`` returns the deterministic Cosmos latent distribution mode.
- ``pool_decoder_input`` applies temporal-mean pooling or preserves latent time.
- ``reconstruct_and_extract`` returns the latent, decoder input, and reconstruction from a compatible VAE.
- ``bcthw_to_uint8`` converts one normalized BCTHW reconstruction to uint8 THWC frames.
- ``save_mp4`` writes visual-only H.264 video frames.
- ``output_name`` creates a readable collision-resistant directory name for a source video.
- ``parse_args`` and ``main`` implement the ``gpi-extract-video`` command-line interface.
