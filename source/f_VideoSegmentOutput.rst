.. _ref_VideoSegmentOutput:

VideoSegmentOutput
==================

Description
-----------

``VideoSegmentOutput`` is the frozen dataclass returned for every file segment processed by :ref:`ref_extract_videos` or ``CosmosVideoExtractor.process_video``. It identifies the segment and the files written for it.

Attributes
----------

- ``video_path`` (*pathlib.Path*): source video path.
- ``segment_index`` (*int*): zero-based segment index.
- ``representation_path`` (*pathlib.Path*): saved ``.pt`` payload.
- ``reconstruction_path`` (*pathlib.Path* or *None*): saved MP4 path, or ``None`` when reconstruction was not requested.

Example Usage
-------------

.. code-block:: python

   outputs = extract_videos(
       videos="input.mp4",
       output_hidden_dir="outputs/hidden",
       segment_seconds=5,
   )

   for output in outputs:
       print(output.segment_index, output.representation_path)
