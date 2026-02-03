#!/usr/bin/env python3
"""
Start the SilkyVoice API server. Pages will be available at:
  - http://127.0.0.1:8000/           (transcription)
  - http://127.0.0.1:8000/tts (multilingual text-to-speech)
"""
import os
import sys

# Optional: avoid OpenMP "already initialized" errors when using PyTorch/NumPy together
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _apply_tts_transformers_patch():
    """Register BeamSearchScorer in transformers LazyModule so Coqui TTS can import it."""
    try:
        import transformers  # noqa: F401
        if hasattr(transformers, "_class_to_module"):
            transformers._class_to_module["BeamSearchScorer"] = "generation.beam_search"
        if hasattr(transformers, "_objects"):
            from transformers.generation.beam_search import BeamSearchScorer
            transformers._objects["BeamSearchScorer"] = BeamSearchScorer
        if hasattr(transformers, "__all__") and "BeamSearchScorer" not in transformers.__all__:
            transformers.__all__.append("BeamSearchScorer")
    except Exception:
        pass


def main():
    _apply_tts_transformers_patch()
    import uvicorn
    host = "127.0.0.1"
    port = 8000
    print(f"Starting SilkyVoice API at http://{host}:{port}")
    print(f"  Transcription:    http://{host}:{port}/")
    print(f"  TTS (Arabic & English): http://{host}:{port}/tts")
    print("Press Ctrl+C to stop.")
    # Use reload=False so long transcription requests (1–2 min first load) don't hang
    use_reload = os.environ.get("SILKYVOICE_RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run("api:app", host=host, port=port, reload=use_reload)

if __name__ == "__main__":
    main()
    sys.exit(0)
