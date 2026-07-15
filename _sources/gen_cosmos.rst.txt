.. _generate_videos:

Generating Videos using Cosmos Tokenizer
========================================

If you want to use GPI for video data, you need to regenerate videos and extract the internal representations of a deep generative model. This section describes how to do so using the `NVIDIA Cosmos Tokenizer <https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-CV8x8x8>`_. In this workflow, generating a video means reconstructing an existing video through the tokenizer.

.. note::
   For data generation, we recommend users to use GPUs. See :ref:`gpu_usage_section`.

.. note::
   This part is based on our paper `Causal Inference with Video Features as Treatments <https://arxiv.org/abs/2607.06126>`_. Please refer to the paper for the application examples and technical details.

What is Cosmos Tokenizer?
-------------------------

Cosmos Tokenizer is a generative model that converts a video into a continuous latent representation and then decodes this representation to reconstruct the video. It has two main components:

1. **Encoder**: The encoder compresses the input frames into a low-dimensional latent tensor.

2. **Decoder**: The decoder converts the latent tensor back into video frames.

For GPI, we use the final tensor immediately before the decoder as the internal representation. By default, **gpi_pack** averages this tensor over the latent-time dimension. The representation has shape ``[B, C, H, W]`` in memory, where ``B`` is the batch size, and it is saved without the batch dimension as ``[C, H, W]``.

Installing the Video Dependencies
---------------------------------

Install the optional packages for reading and writing videos:

.. code-block:: bash

   python -m pip install --upgrade "gpi-pack[video]"

Version 0.2.1 and its video dependencies are available from PyPI.

The default Cosmos checkpoint requires you to accept its access conditions on Hugging Face. If the checkpoint is not already available in your cache, log in to Hugging Face before running the example.

The upstream model card specifies at most 121 input frames and a minimum
shorter-side resolution of 256 pixels. ``extract_videos`` applies the
121-frame default, but it does not enforce the spatial minimum; choose
``frame_size`` accordingly. NVIDIA reports BF16 validation on Ampere and
Hopper GPUs. The package also permits FP32 and CPU execution, but those
configurations do not have the same upstream hardware validation and are
usually much slower.

How to use Cosmos Tokenizer
---------------------------

The ``extract_videos`` function provides a simple interface for segmenting videos, regenerating each segment, and saving its internal representation. The input can be one video file, one directory, or a list containing files and directories.

.. code-block:: python

   import torch
   from gpi_pack.video import extract_videos

   outputs = extract_videos(
       videos="path/to/videos",                  # one file, directory, or list
       output_hidden_dir="outputs/hidden",
       output_video_dir="outputs/reconstructed", # omit to skip reconstruction
       segment_seconds=5,
       frame_size=(320, 480),
       temporal_pooling="temporal_mean", # pooling strategy for the latent time dimension
   )

   first_output = outputs[0]
   payload = torch.load(
       first_output.representation_path,
       map_location="cpu",
       weights_only=True,
   )
   representation = payload["representation"]   # [C, H, W]

Each input video is divided into consecutive segments of ``segment_seconds`` seconds. The function saves one ``.pt`` file for each segment and returns the output paths in segment order.

If you only need the internal representations, omit ``output_video_dir``. This skips the decoder and is faster:

.. code-block:: python

   outputs = extract_videos(
       videos="path/to/videos",
       output_hidden_dir="outputs/hidden",
       segment_seconds=5,
   )



Arguments
---------

The function ``extract_videos`` has the following arguments:

- ``videos``: a supported video file, a directory searched recursively, or a list of files and directories (**required**). Supported extensions are ``.mp4``, ``.mov``, ``.avi``, ``.mkv``, ``.webm``, and ``.m4v``.

- ``output_hidden_dir``: root directory for the saved representation files (**required**).

- ``segment_seconds``: positive duration of each video segment in seconds (**required**).

- ``output_video_dir``: directory for reconstructed MP4 files. If ``None``, the function does not run the decoder or save reconstructions. The default is ``None``.

- ``segment``: optional zero-based segment index or inclusive ``(start, end)`` tuple. Use ``(start, None)`` for an open-ended range.

- ``model_id``: Hugging Face model identifier. The default is ``"nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"``.

- ``device``: device used for inference. If ``None``, the function uses CUDA when available and otherwise uses CPU.

- ``cache_dir``: directory for caching the model files. The default is the Hugging Face cache.

- ``dtype``: model precision. Supported strings are ``"auto"``, ``"bf16"``, ``"fp16"``, and ``"fp32"``. The automatic setting uses BF16 on a compatible CUDA GPU and FP32 otherwise.

- ``frame_size``: optional ``(height, width)`` used to resize every frame. No resizing is applied by default. Use a common size when you need to stack representations from videos with different resolutions.

- ``max_frames``: maximum number of frames passed to the model for one segment. Longer segments are sampled uniformly; both endpoints are retained when at least two frames are selected. The default is 121; use ``None`` to disable sampling.

- ``pad_multiple``: spatial padding multiple required by the model. If ``None``, the function uses the VAE's spatial compression ratio, which is normally 8.

- ``temporal_pooling``: ``"temporal_mean"`` averages over latent time and saves ``[C, H, W]``. ``"none"`` retains latent time and saves ``[C, D, H, W]``. The default is ``"temporal_mean"``.

- ``drop_last``: whether to discard the final segment when it is shorter than ``segment_seconds``. The default is ``False``.

- ``save_latent``: whether to include the encoder latent in each payload. The default is ``False``.

- ``save_decoder_input``: whether to include the unpooled decoder input in each payload. The default is ``False``.

- ``overwrite``: whether to replace existing output files. The default is ``False``.

- ``extractor``: an existing ``CosmosVideoExtractor`` to reuse across calls. If provided, its model settings are used.

- ``pretrained_kwargs``: extra options passed to ``from_pretrained``, such as ``token``, ``revision``, or ``local_files_only``.

- ``verbose``: whether to print processing progress. The default is ``True``.

Output Files
------------

The saved ``.pt`` payload contains ``representation`` as an unbatched CPU ``float32`` tensor. It also records the model identifier and revision, library versions, segment boundaries, frame rate, selected frame indices, input and padded shapes, resizing settings, pooling settings, and representation shape. If requested, it also contains ``latent`` and ``decoder_input``.

The output files are named ``segment_000000.pt``, ``segment_000001.pt``, and so on inside a separate directory for each source video. Reconstructed files use names such as ``segment_000000_recon.mp4``.

Command-Line Interface
----------------------

The same pipeline is available from the command line:

.. code-block:: bash

   gpi-extract-video \
       --input path/to/videos \
       --output_dir outputs \
       --segment_seconds 5 \
       --frame_size 320 480 \
       --save_reconstruction

Important Notes
---------------

- Audio is not encoded and is not copied into reconstructed videos. To transcribe the original audio separately, see :ref:`transcribe_audio`.
- Reconstructed MP4 files use H.264 and require an even frame height and width.
- Variable-frame-rate videos are segmented using their nominal frame rate.
- The input directory and output directory must be different.
- Existing files raise an error unless ``overwrite=True`` is supplied.

For the complete API, see :ref:`ref_extract_videos` and :ref:`ref_CosmosVideoExtractor`.
