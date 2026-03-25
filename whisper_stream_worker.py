#!/usr/bin/env python3
"""
Worker process for *streaming* Whisper transcription.

It reads line-delimited messages from stdin:
  - <path_to_chunk_webm>  (one line per chunk)
  - END                  (terminate)

For each received chunk it:
  - converts webm -> 16kHz mono wav (via ffmpeg)
  - runs faster-whisper on that chunk
  - emits a JSON line to stdout with partial + full transcript so far

Output format (one JSON object per line):
  { "type": "partial", "chunk_index": 0, "text": "...", "full_text": "...", ... }
  { "type": "final",   "full_text": "...", ... }
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def _ffmpeg_to_wav_16k_mono(input_path: Path, wav_path: Path) -> None:
    """
    Convert an incoming chunk into 16kHz mono WAV.

    Note: some MediaRecorder chunks might not be independently decodable.
    When that happens, ffmpeg will fail and we raise with stderr for debugging.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg failed (exit={proc.returncode}) while converting chunk: {input_path.name}. "
            f"stderr={stderr[:2000]!r}"
        )


def main() -> None:
    # Use env to allow overriding in deployments.
    model_dir = os.environ.get("SILKYVOICE_WHISPER_MODEL_DIR", "./models/faster-whisper-large-v3")
    device = os.environ.get("SILKYVOICE_WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("SILKYVOICE_WHISPER_COMPUTE_TYPE", "int8")

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(json.dumps({"type": "error", "detail": f"Import error: {e}"}), flush=True)
        sys.exit(1)

    model = WhisperModel(model_dir, device=device, compute_type=compute_type, local_files_only=True)

    full_text = ""
    detected_language = None  # filled after the first chunk (if we start with language='auto')
    chunk_index = 0
    last_language_probability = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == "END":
            break

        # Line might be a JSON config in the future; ignore for now.
        if line.startswith("{") and line.endswith("}"):
            continue

        webm_path = Path(line)
        try:
            if not webm_path.exists():
                raise FileNotFoundError(str(webm_path))

            wav_path = webm_path.with_suffix(".wav")
            transcribe_path = None
            # Prefer converting to wav for consistent decode/resampling.
            try:
                _ffmpeg_to_wav_16k_mono(webm_path, wav_path)
                transcribe_path = str(wav_path)
            except Exception:
                # Fallback: some chunks may not be ffmpeg-decodable as WAV;
                # faster-whisper's decoder may still succeed on the original container.
                transcribe_path = str(webm_path)

            # For speed: smaller beam size. For realtime: chunk-by-chunk.
            # Language: if detected_language is known, reuse it to avoid per-chunk detection overhead.
            language_param = detected_language
            segments, info = model.transcribe(
                transcribe_path,
                language=language_param,
                beam_size=1,
                best_of=1,
                vad_filter=True,
            )

            if detected_language is None:
                detected_language = info.language
                last_language_probability = info.language_probability
            else:
                # Keep probability from first detection (more stable).
                last_language_probability = last_language_probability

            chunk_text = "".join(seg.text for seg in segments).strip()

            if chunk_text:
                if full_text and not full_text.endswith((" ", "\n")):
                    full_text += " "
                full_text += chunk_text

            out = {
                "type": "partial",
                "chunk_index": chunk_index,
                "language": detected_language,
                "language_probability": last_language_probability,
                "text": chunk_text,
                "full_text": full_text.strip(),
            }
            print(json.dumps(out, ensure_ascii=False), flush=True)

            # Cleanup to keep long recordings from accumulating files.
            try:
                webm_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

            chunk_index += 1
        except Exception as e:
            out = {"type": "error", "chunk_index": chunk_index, "detail": str(e)}
            print(json.dumps(out, ensure_ascii=False), flush=True)
            chunk_index += 1

    print(
        json.dumps(
            {
                "type": "final",
                "language": detected_language,
                "language_probability": last_language_probability,
                "full_text": full_text.strip(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

