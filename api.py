import json
import logging
import os
import subprocess
import sys
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel
import tempfile
import shutil
from typing import Optional

from multilingual_tts import synthesize as tts_synthesize

logger = logging.getLogger("uvicorn.error")
app = FastAPI()

# Whisper runs in a separate process to isolate segfaults (faster-whisper/ctranslate2 on macOS).
_WHISPER_WORKER = Path(__file__).resolve().parent / "whisper_worker.py"
_WHISPER_STREAM_WORKER = Path(__file__).resolve().parent / "whisper_stream_worker.py"
_TRANSCRIBE_TIMEOUT = 300
_SUBPROCESS_ENV = {**os.environ, "OMP_NUM_THREADS": "1"}


class TTSBody(BaseModel):
    text: str
    language: Optional[str] = "auto"  # "auto" | "ar" | "en"

# Allow your HTML page (e.g. http://localhost:8000 or file://) to call the API.
# For local development we allow all origins; tighten this for production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health endpoint for GET requests.
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index_page():
    """Serve the main transcription page."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# Content types we explicitly accept; others (e.g. empty, application/octet-stream) are still attempted
_ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/webm",
    "audio/ogg", "audio/x-m4a", "audio/mp4",
}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accept an uploaded audio file and return Whisper transcription.
    Runs Whisper in a separate process to isolate segfaults; server stays up if the worker crashes.
    """
    logger.info("POST /transcribe: request received")
    ct = (file.content_type or "").strip()
    if ct and ct not in _ALLOWED_AUDIO_TYPES and not ct.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use an audio file (e.g. WAV, MP3, WebM).",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            with file.file as src:
                shutil.copyfileobj(src, tmp)
            tmp_path = tmp.name

        size = Path(tmp_path).stat().st_size
        if size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty. Record again or choose a non-empty file.")
        if size < 500:
            raise HTTPException(status_code=400, detail="Audio file is too small (likely empty or broken recording).")

        logger.info("POST /transcribe: file saved (%s bytes), starting Whisper worker process...", size)
        proc = subprocess.run(
            [sys.executable, str(_WHISPER_WORKER), tmp_path],
            capture_output=True,
            text=True,
            timeout=_TRANSCRIBE_TIMEOUT,
            env=_SUBPROCESS_ENV,
            cwd=Path(__file__).resolve().parent,
        )
        if proc.returncode != 0:
            msg = proc.stderr or f"Worker exited with code {proc.returncode}"
            logger.error("POST /transcribe: worker failed: %s", msg)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {msg}")

        out = json.loads(proc.stdout)
        logger.info("POST /transcribe: done")
        return out
    except subprocess.TimeoutExpired:
        logger.error("POST /transcribe: worker timed out")
        raise HTTPException(status_code=504, detail="Transcription timed out.")
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.exception("POST /transcribe: invalid worker output: %s", e)
        raise HTTPException(status_code=500, detail="Transcription produced invalid output.")
    except Exception as e:
        logger.exception("POST /transcribe failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


@app.websocket("/transcribe/stream")
async def transcribe_stream(websocket: WebSocket):
    """
    Streaming transcription over WebSocket.

    Client protocol:
      - send binary websocket messages: each message is an audio chunk (e.g. WebM from MediaRecorder)
      - send text message "end" to finalize
    Server protocol:
      - emits JSON lines as text frames:
          {type:"partial", chunk_index, text, full_text, language, language_probability}
          {type:"final", full_text, language, language_probability}
          {type:"error", detail}
    """
    logger.info("WS /transcribe/stream: connection attempt from %s", websocket.client)
    await websocket.accept()
    logger.info("WS /transcribe/stream: accepted")

    if not _WHISPER_STREAM_WORKER.exists():
        await websocket.send_text(json.dumps({"type": "error", "detail": "whisper_stream_worker.py not found"}))
        await websocket.close()
        return

    # Use a dedicated temp directory per websocket session.
    tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_stream_"))
    chunk_index = 0
    proc: Optional[asyncio.subprocess.Process] = None
    stdout_task: Optional[asyncio.Task] = None

    async def _start_worker() -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            str(_WHISPER_STREAM_WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_SUBPROCESS_ENV,
            cwd=Path(__file__).resolve().parent,
        )

    async def _worker_stdout_forward() -> None:
        # Worker prints JSON lines; forward each line to client.
        assert proc is not None and proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                await websocket.send_text(line.decode("utf-8", errors="replace").rstrip("\n"))
            except Exception:
                break

    try:
        proc = await _start_worker()
        stdout_task = asyncio.create_task(_worker_stdout_forward())

        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # Text frames: use "end" to finalize.
            if msg.get("text") is not None:
                text_msg = (msg.get("text") or "").strip().lower()
                if text_msg in ("end", "stop", "final", "done"):
                    break
                continue

            # Binary frames are treated as audio chunks.
            chunk_bytes = msg.get("bytes")
            if not chunk_bytes:
                continue
            # FastAPI may expose binary frames as `memoryview`; normalize to `bytes`.
            if isinstance(chunk_bytes, memoryview):
                chunk_bytes = chunk_bytes.tobytes()

            webm_path = tmp_dir / f"chunk_{chunk_index}.webm"
            webm_path.write_bytes(chunk_bytes)

            assert proc.stdin is not None
            proc.stdin.write((str(webm_path) + "\n").encode("utf-8"))
            await proc.stdin.drain()

            chunk_index += 1

        # Signal worker end and wait briefly for it to finish.
        if proc and proc.stdin:
            try:
                proc.stdin.write(b"END\n")
                await proc.stdin.drain()
            except Exception:
                pass

        # Ensure we forward whatever is left from stdout.
        if stdout_task:
            try:
                await asyncio.wait_for(stdout_task, timeout=20.0)
            except asyncio.TimeoutError:
                stdout_task.cancel()

        if proc:
            try:
                await asyncio.wait_for(proc.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                proc.kill()
        logger.info("WS /transcribe/stream: finished (chunks=%s)", chunk_index)

    except WebSocketDisconnect:
        # Client disconnected; try to shutdown worker.
        try:
            if proc and proc.stdin:
                proc.stdin.write(b"END\n")
                await proc.stdin.drain()
        except Exception:
            pass
        logger.info("WS /transcribe/stream: client disconnected (chunks=%s)", chunk_index)
    except Exception as e:
        logger.exception("transcribe_stream failed: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(e)}))
        except Exception:
            pass
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# --- Multilingual Text-to-Speech (Arabic & English) ---

def _normalize_lang(lang: str) -> str:
    if not lang or not lang.strip():
        return "auto"
    l = lang.strip().lower()
    if l in ("auto", "ar", "en", "arabic", "english"):
        return "ar" if l == "arabic" else "en" if l == "english" else l
    return "auto"


@app.post("/tts", response_class=Response)
async def tts_speak(body: TTSBody):
    """
    Synthesize text to speech (Arabic or English). Send JSON: {"text": "...", "language": "auto"|"ar"|"en"}.
    Returns WAV audio.
    """
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Missing or empty 'text' field")
    lang = _normalize_lang(body.language or "auto")
    if lang not in ("auto", "ar", "en"):
        lang = "auto"
    try:
        wav_bytes, _ = tts_synthesize(body.text.strip(), language=lang)
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/arabic", response_class=Response)
async def tts_arabic(body: TTSBody):
    """
    Synthesize Arabic text to speech (language forced to Arabic). Send JSON: {"text": "نص عربي"}.
    Returns WAV audio.
    """
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Missing or empty 'text' field")
    try:
        wav_bytes, _ = tts_synthesize(body.text.strip(), language="ar")
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts", response_class=HTMLResponse)
async def tts_page():
    """Serve the multilingual TTS web interface (Arabic & English)."""
    html_path = Path(__file__).parent / "arabic_tts.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="arabic_tts.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/arabic-tts", response_class=RedirectResponse)
async def arabic_tts_redirect():
    """Redirect legacy URL to /tts."""
    return RedirectResponse(url="/tts", status_code=301)
