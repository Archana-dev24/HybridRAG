"""Backward-compatible entry point. Use pipeline.py directly for new code."""

from pipeline import HybridRAGPipeline, main  # noqa: F401

if __name__ == "__main__":
    main()
