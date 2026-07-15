.. _ref_VideoExtractionResult:

VideoExtractionResult
=====================

Description
-----------

``VideoExtractionResult`` is the frozen dataclass returned by the in-memory methods of :ref:`ref_CosmosVideoExtractor`. All tensors retain their batch dimension.

Attributes
----------

- ``latent`` (*torch.Tensor*): deterministic encoder latent.
- ``decoder_input`` (*torch.Tensor*): unpooled tensor immediately before the decoder.
- ``representation`` (*torch.Tensor*): ``[B, C, H, W]`` with temporal-mean pooling or ``[B, C, D, H, W]`` without pooling.
- ``reconstruction`` (*torch.Tensor* or *None*): reconstructed BCTHW video, or ``None`` when only encoding was requested.
- ``input_shape_bcthw`` (*tuple*): shape after normalization and optional resizing.
- ``padded_shape_bcthw`` (*tuple*): model-input shape after spatial padding.
- ``pad_bottom`` and ``pad_right`` (*int*): padding applied to the bottom and right edges.
- ``original_num_frames`` (*int*): number of frames before sampling.
- ``selected_frame_indices`` (*tuple of int*): original indices retained for the model.
- ``temporal_pooling`` (*str*): pooling rule used to create ``representation``.
- ``pre_decoder_hidden`` (*torch.Tensor*): compatibility property that returns ``decoder_input``.

Example Usage
-------------

.. code-block:: python

   result = extractor.encode_video(frames)
   representation = result.representation
   selected = result.selected_frame_indices
