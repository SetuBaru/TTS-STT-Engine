#!/usr/bin/env python3
"""
Start the SilkyVoice API server. Pages will be available at:
  - http://192.168.8.123:8231          (transcription)
  - http://192.168.8.123:8231/tts (multilingual text-to-speech)
"""
import os
import sys

# Optional: avoid OpenMP "already initialized" errors when using PyTorch/NumPy together
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    import uvicorn
    host = "192.168.8.123"
    port = 8231
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
