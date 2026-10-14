"""Reference baselines: TypeSafe's Jev and Gemini on Vertex, run over the same test set.

Both clients read ``data/*.jsonl`` (``DataRecord``) and write ``predictions.jsonl``
(``Prediction``), ``cost.json`` (``Cost``) and ``summary.json`` to ``--out``.
"""
