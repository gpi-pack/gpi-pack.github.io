.. _transcribe_audio:

Transcribing Audio using Whisper
================================

Videos often contain spoken information that is not represented by the image frames alone. This section describes how to transcribe that information using `Whisper <https://github.com/openai/whisper>`_, an open-source automatic speech-recognition model developed by OpenAI. The resulting transcripts can be aligned with the video segments and used to construct the text representations in a Video-as-Treatment analysis.

.. note::
   Whisper is a separate third-party package and is not installed with **gpi_pack**. The Cosmos video extractor does not encode audio or copy it into reconstructed videos, so you should transcribe the original media file or an audio file extracted from it.

.. note::
   Whisper can run on a CPU, but a CUDA-capable GPU is recommended for faster transcription, particularly with the larger models.

What is Whisper?
----------------

Whisper is an encoder-decoder Transformer. The original model series described
in the Whisper paper was trained on 680,000 hours of multilingual and multitask
audio data; the repository now also includes later ``large-v2``, ``large-v3``,
and ``turbo`` releases. Whisper converts audio into a log-Mel spectrogram,
encodes the acoustic information, and predicts text tokens with a decoder. It
can transcribe speech in its original language, identify the spoken language,
and, with supported multilingual models, translate speech into English. See
the `Whisper paper <https://arxiv.org/abs/2212.04356>`_ and the official model
card for the release details.

The high-level ``transcribe`` method processes a complete media file using a sliding 30-second window. It returns the complete text together with timestamped segments and other decoding information.

Installing Whisper
------------------

Install the latest released Whisper package in the same Python environment used for your analysis:

.. code-block:: bash

   python -m pip install -U openai-whisper

Whisper also requires the ``ffmpeg`` command-line program to read audio. For example, install it with Homebrew on macOS or APT on Ubuntu and Debian:

.. code-block:: bash

   # macOS
   brew install ffmpeg

   # Ubuntu or Debian
   sudo apt update
   sudo apt install ffmpeg

Confirm that ``ffmpeg`` is available on your command path:

.. code-block:: bash

   ffmpeg -version

Whisper downloads the selected model weights the first time you load that model and stores them in the local cache.

Choosing a Whisper Model
------------------------

Whisper provides several model sizes. Larger models generally require more memory and computation, while smaller models are faster.

- ``turbo`` is an optimized version of ``large-v3`` and is a useful default for multilingual transcription.
- ``tiny``, ``base``, ``small``, and ``medium`` provide a range of speed and accuracy tradeoffs. English-only variants such as ``base.en`` are also available.
- ``large`` provides the largest multilingual model family but requires more memory and computation.

The ``turbo`` model is intended for transcription and does not perform speech translation reliably. To translate non-English speech into English, use a multilingual model such as ``medium`` or ``large`` with the translation task.

Command-Line Usage
------------------

The ``whisper`` command can transcribe one or more audio files. The following command uses the ``turbo`` model and writes all supported output formats to ``outputs/transcripts``:

.. code-block:: bash

   whisper path/to/audio.mp3 \
       --model turbo \
       --output_dir outputs/transcripts \
       --output_format all

The available output formats include plain text, JSON, SRT, VTT, and TSV. Whisper detects the spoken language automatically when ``--language`` is omitted. You can specify it when it is known:

.. code-block:: bash

   whisper japanese.wav \
       --model turbo \
       --language Japanese \
       --task transcribe

To translate the same recording into English, use a multilingual model that supports translation:

.. code-block:: bash

   whisper japanese.wav \
       --model medium \
       --language Japanese \
       --task translate

Run ``whisper --help`` to see the complete list of decoding and output options.

Python Usage
------------

The Python interface is useful when you want to keep the transcript and its timestamps in the same analysis script:

.. code-block:: python

   import whisper

   model = whisper.load_model("turbo")
   result = model.transcribe(
       "path/to/audio.mp3",
       task="transcribe",
       word_timestamps=True,
       verbose=False,
   )

   print(result["text"])

The ``segments`` entry contains start and end times in seconds:

.. code-block:: python

   for segment in result["segments"]:
       print(
           f'{segment["start"]:.2f}--{segment["end"]:.2f}: '
           f'{segment["text"].strip()}'
       )

When ``word_timestamps=True``, every segment also contains a ``words`` list with approximate word-level start and end times.

Transcribing Audio from a Video
-------------------------------

Use the original video rather than a Cosmos reconstruction, because the
reconstructed files do not contain audio. Whisper can read the original video
directly through ffmpeg:

.. code-block:: python

   result = model.transcribe(
       "path/to/source_video.mp4",
       task="transcribe",
       word_timestamps=True,
       verbose=False,
   )

You can alternatively extract its audio track first:

.. code-block:: bash

   ffmpeg -i path/to/source_video.mp4 \
       -vn \
       path/to/source_audio.wav

Then pass ``source_audio.wav`` to either the command-line or Python example above.

Aligning Transcripts with Video Segments
----------------------------------------

For Video-as-Treatment, align timestamps to the segment metadata written by
Cosmos. This is more precise than calculating ``timestamp // segment_seconds``
because the package converts the requested duration to an integer number of
frames using the video's nominal frame rate. Here, ``outputs`` is the list
returned by ``extract_videos`` for the same source video.

.. code-block:: python

   import torch

   boundaries = []
   for output in outputs:
       payload = torch.load(
           output.representation_path,
           map_location="cpu",
           weights_only=True,
       )
       boundaries.append(
           (
               float(payload["start_time_sec"]),
               float(payload["end_time_sec"]),
           )
       )

   words_by_segment = [[] for _ in outputs]

   for whisper_segment in result["segments"]:
       for word in whisper_segment.get("words", []):
           midpoint = (float(word["start"]) + float(word["end"])) / 2
           for segment_id, (start, end) in enumerate(boundaries):
               if start <= midpoint < end:
                   words_by_segment[segment_id].append(word["word"])
                   break

   transcript_segments = [
       "".join(words).strip()
       for words in words_by_segment
   ]

The resulting ``transcript_segments`` list follows the same order as the video representations. You can pass these texts through your text representation model to construct the aligned ``R_text`` input described in :ref:`ref_VideoAsTreatment`.

.. note::
   Timestamp-based alignment is approximate. An utterance can cross a video-segment boundary, so inspect the alignment and choose a consistent assignment rule before constructing the analysis data.

Important Notes
---------------

- Transcription accuracy depends on language, accent, recording quality, background noise, and overlapping speech. Review important transcripts before using them to define treatments, outcomes, or confounders.
- Whisper does not perform speaker diarization, so it does not identify which person produced each utterance.
- Word and segment timestamps are estimates and should not be treated as exact annotations.
- The open-source Whisper package performs inference locally after downloading the model weights; make sure your storage and computing environment are appropriate for sensitive audio.
- Model size should be treated as a preprocessing choice. Use the same model and decoding settings across observations and record them for reproducibility.

For additional options and the current list of models, see the `official Whisper repository <https://github.com/openai/whisper>`_.
