.. _ref_load_hiddens:

load_hiddens
============

Description
-----------

The ``load_hiddens`` function loads tensor-only ``.pt`` representation files in the order supplied by the user. It stacks the tensors, removes a singleton second dimension when present, converts the result to CPU float values, and returns a NumPy array. It is intended for LLM and image files that contain a tensor directly.

Arguments
---------

- ``directory`` (*str*): directory containing the files.
- ``hidden_list`` (*list*): file identifiers in the required output order, without ``.pt``.
- ``prefix`` (*str*, optional): filename prefix. If ``None``, each identifier is used directly.
- ``device`` (*torch.device* or *str*, optional): map location used while loading. The default is ``"cpu"``.

Returns
-------

- *np.ndarray*: stacked representations.

Example Usage
-------------

.. code-block:: python

   from gpi_pack.TarNet import load_hiddens

   hidden_states = load_hiddens(
       directory="outputs/hidden",
       hidden_list=df.index.tolist(),
       prefix="hidden_",
   )

Missing files raise ``FileNotFoundError``.

.. note::
   :ref:`ref_extract_videos` saves a metadata dictionary rather than a bare tensor. Load a video payload with ``torch.load`` and read its ``payload["representation"]`` field instead of using this function.
