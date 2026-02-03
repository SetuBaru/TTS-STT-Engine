# SilkyVoice – Transcription & Multilingual Text-to-Speech API

SilkyVoice is a FastAPI backend that provides:

- **Speech-to-Text**: Upload or record audio and get transcriptions using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- **Text-to-Speech**: Generate speech from Arabic or English text using Coqui XTTS-v2 (multilingual TTS).

Web UIs are served at `/` (transcription) and `/tts` (TTS).

---

## Project structure

- `api.py` – FastAPI app (transcription + TTS endpoints).
- `index.html` – transcription UI (upload/record audio, view result).
- `arabic_tts.html` – TTS UI (enter text, choose language, generate audio); served at `/tts`.
- `main.py` – one-off transcription script using the local Whisper model.
- `run_server.py` – start the API server (recommended).
- `multilingual_tts.py` – Coqui XTTS-v2 TTS (Arabic & English).
- `models/large-v3/` – local Whisper model directory.
- `test.wav` – example audio file (optional).

---

## Prerequisites

- Python 3.9+ (recommended)
- A working C++ toolchain (required by some `faster-whisper` dependencies on some platforms)
- **Transcription**: Whisper model at `./models/large-v3` (download a compatible model for `faster-whisper` if needed).
- **TTS**: Coqui XTTS-v2 is downloaded automatically on first use (multilingual Arabic & English).

---

## Installation

1. **Clone or copy the project directory** (this repo).

2. **Create and activate a virtual environment** (recommended):

   ```bash
   cd SilkyVoice
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model is in place** at `./models/large-v3`. Your tree should look roughly like:

   ```text
   SilkyVoice/
     api.py
     index.html
     main.py
     models/
       large-v3/
         config.json
         model.bin
         preprocessor_config.json
         tokenizer.json
         vocabulary.json
         ...
   ```

---

## Running the API server

From the project root:

```bash
python run_server.py
```

Or with the venv:

```bash
.venv/bin/python run_server.py
```

The server starts at **http://127.0.0.1:8000**. Pages:

- **Transcription**: http://127.0.0.1:8000/
- **Text-to-Speech**: http://127.0.0.1:8000/tts

Alternatively, run with Uvicorn directly:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

---

## API documentation

Base URL: **http://127.0.0.1:8000** (or your host/port).

Interactive docs: **http://127.0.0.1:8000/docs** (Swagger UI).

### Health

| Method | Path     | Description        |
|--------|----------|--------------------|
| `GET`  | `/health`| Health check       |

**Example**

```bash
curl http://127.0.0.1:8000/health
```

**Response (200)**

```json
{ "status": "ok" }
```

---

### Transcription (speech-to-text)

| Method | Path         | Description                    |
|--------|--------------|--------------------------------|
| `POST` | `/transcribe`| Upload audio, get transcription|

**Request**

- **Content-Type**: `multipart/form-data`
- **Body**: form field `file` = audio file

**Supported audio types**: `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp3`, `audio/webm`, `audio/ogg`, `audio/x-m4a`, `audio/mp4` (or any `audio/*`).

**Example**

```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
     -F "file=@test.wav"
```

**Response (200, JSON)**

```json
{
  "language": "en",
  "language_probability": 0.98,
  "text": "Transcribed text here..."
}
```

---

### Text-to-Speech (multilingual, Arabic & English)

| Method | Path        | Description                              |
|--------|-------------|------------------------------------------|
| `GET`  | `/tts`      | Serve TTS web UI                         |
| `POST` | `/tts`      | Synthesize speech from text (JSON body) |
| `POST` | `/tts/arabic` | Same as `POST /tts` with language forced to Arabic |

**Request (POST /tts)**

- **Content-Type**: `application/json`
- **Body**:
  - `text` (string, required): text to synthesize (Arabic or English).
  - `language` (string, optional): `"auto"` (detect from text), `"ar"` (Arabic), or `"en"` (English). Default: `"auto"`.

**Example (English)**

```bash
curl -X POST "http://127.0.0.1:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "language": "en"}' \
     --output speech.wav
```

**Example (Arabic)**

```bash
curl -X POST "http://127.0.0.1:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "مرحبا بك", "language": "ar"}' \
     --output speech.wav
```

**Example (auto-detect language)**

```bash
curl -X POST "http://127.0.0.1:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world or مرحبا"}' \
     --output speech.wav
```

**Response (200)**

- **Content-Type**: `audio/wav`
- **Body**: raw WAV audio bytes (e.g. save to a `.wav` file).

**Errors**

- **400**: missing or empty `text`, or validation error.
- **500**: server error (e.g. TTS model failure).

---

## Using the web UI

1. **Start the API server** (see [Running the API server](#running-the-api-server)).

2. **Transcription** – open **http://127.0.0.1:8000/** (or the root URL of your server):
   - Upload an audio file and click **"Transcribe Uploaded File"**, or
   - Click **"Start Recording"**, speak, then **"Stop & Transcribe"**.
   - The transcription and detected language appear below (e.g. double-click it, or open it via your browser’s “File → Open” dialog).

3. **Text-to-Speech** – open **http://127.0.0.1:8000/tts**:
   - Choose **Language** (Auto, Arabic, or English).
   - Enter text in the **Text** field (Arabic or English).
   - Click **"Generate audio"** and use the player to listen.

The UIs use the same host/port as the server (e.g. `http://127.0.0.1:8000`). If you run the server on a different host or port, open the corresponding URLs.

---

## One-off transcription script (`main.py`)

The `main.py` file demonstrates a simple, non-API usage of the model:

```python
from faster_whisper import WhisperModel

model = WhisperModel("./models/large-v3", device="cpu", compute_type="int8")

segments, info = model.transcribe("test.wav")

for s in segments:
    print(s.text)
```

To run it:

```bash
python main.py
```

This will transcribe `test.wav` (if present) and print the text to the console.

---

## Notes and tips

- **Performance**: The `compute_type="int8"` and `device="cpu"` settings are suitable for CPU-only machines. If you have a GPU, you can adjust these parameters in both `api.py` and `main.py` for better throughput.
- **CORS**: `api.py` currently allows all origins (via `allow_origins=["*"]`) for ease of local development. For production, restrict this to trusted origins only.
- **Error handling**: The API returns structured errors (HTTP 400/500 with a `detail` message) if the file type is unsupported or something goes wrong during transcription.

