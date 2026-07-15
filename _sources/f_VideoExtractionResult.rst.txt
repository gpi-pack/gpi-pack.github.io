.. _ref_VideoExtractionResult:

VideoExtractionResult
=====================

Description
-----------

``VideoExtractionResult`` is the frozen dataclass returned by the in-memory methods of :ref:`ref_CosmosVideoExtractor`. All tensors retain their batch dimension.

.. code-block:: python

   VideoExtractionResult(
       latent,
       decoder_input,
       representation,
       reconstruction,
       input_shape_bcthw,
       padded_shape_bcthw,
       pad_bottom,
       pad_right,
       original_num_frames,
       selected_frame_indices,
       temporal_pooling,
   )

Attributes
----------

- ``latent`` (*torch.Tensor*): deterministic encoder latent with shape ``[B, C_latent, D_latent, H_latent, W_latent]``.
- ``decoder_input`` (*torch.Tensor*): unpooled tensor immediately before the decoder, with shape ``[B, C, D, H, W]``.
- ``representation`` (*torch.Tensor*): ``[B, C, H, W]`` with temporal-mean pooling or ``[B, C, D, H, W]`` without pooling.
- ``reconstruction`` (*torch.Tensor* or *None*): reconstructed ``[B, 3, T, H, W]`` video, or ``None`` when only encoding was requested.
- ``input_shape_bcthw`` (*tuple of int*): shape after frame sampling, normalization, and optional resizing, before spatial padding.
- ``padded_shape_bcthw`` (*tuple*): model-input shape after spatial padding.
- ``pad_bottom`` and ``pad_right`` (*int*): padding applied to the bottom and right edges.
- ``original_num_frames`` (*int*): number of frames before sampling.
- ``selected_frame_indices`` (*tuple of int*): zero-based original frame indices retained for the model.
- ``temporal_pooling`` (*str*): pooling rule used to create ``representation``.
- ``pre_decoder_hidden`` (*torch.Tensor*): read-only compatibility property that returns the same tensor as ``decoder_input``.

``encode_video`` fills every field except ``reconstruction``. ``reconstruct_video`` and ``transform_video`` additionally decode the latent and fill that field.

Example Usage
-------------

.. code-block:: python

   result = extractor.encode_video(frames)
   representation = result.representation
   selected = result.selected_frame_indices
