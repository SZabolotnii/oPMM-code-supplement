"""Dataset download scripts for mv-patp-mimo external benchmarks.

Each ``download_<dataset>.py`` module retrieves a public benchmark, verifies
its SHA-256 hash against ``data/expected_hashes.json``, and writes the
artefact under ``data/<dataset>/``. Downloads are gated by an explicit
``--accept-license`` flag because the underlying datasets carry their own
terms of use (DeepMIMO research licence, WAIR-D research licence,
RadioML CC-BY-NC-SA 4.0).
"""
