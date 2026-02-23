#!/usr/bin/env python3
"""
Standalone worker for Whisper transcription. Run as:
  python whisper_worker.py <path_to_audio_file>

Loads faster-whisper in this process only, so any segfault is isolated from the API server.
Prints JSON result to stdout and exits 0 on success; stderr + non-zero exit on error.
"""
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python whisper_worker.py <audio_file_path>", file=sys.stderr)
        sys.exit(2)
    audio_path = Path(sys.argv[1])
    if not audio_path.is_file():
        print(f"Not a file: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    model = WhisperModel("./models/large-v3", device="cuda", compute_type="int8", local_files_only=True)
    segments, info = model.transcribe(str(audio_path))
    full_text = "".join(segment.text for segment in segments)
    out = {
        "language": info.language,
        "language_probability": info.language_probability,
        "text": full_text.strip(),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
