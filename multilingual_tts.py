"""
Multilingual Text-to-Speech using NAMAA-Saudi-TTS (Chatterbox).
Supports Arabic text-to-speech with Saudi dialect.
"""

import os
import tempfile
from typing import Optional, Literal

import torch
import torchaudio as ta
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from chatterbox import mtl_tts

# PyTorch 2.6+ uses weights_only=True by default in torch.load; allow TTS/Chatterbox
# model config classes so checkpoint loading does not fail.
def _allowlist_tts_globals():
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
    except Exception:
        pass


_allowlist_tts_globals()

# Lazy-loaded model
_model = None
_ckpt_dir = None

# NAMAA-Saudi-TTS sample rate
SAMPLE_RATE = 24000


def _get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _get_tts():
    """Lazy-load the NAMAA-Saudi-TTS model."""
    global _model, _ckpt_dir
    
    if _model is None:
        device = _get_device()
        
        # Download model repo
        _ckpt_dir = snapshot_download(
            repo_id="NAMAA-Space/NAMAA-Saudi-TTS",
            repo_type="model",
            revision="main"
        )
        
        # Load base model
        _model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)
        
        # Load Saudi checkpoint
        t3_path = os.path.join(_ckpt_dir, "t3_mtl23ls_v2.safetensors")
        t3_state = load_safetensors(t3_path, device=device)
        
        _model.t3.load_state_dict(t3_state)
        _model.t3.to(device).eval()
    
    return _model


def _has_arabic_chars(s: str) -> bool:
    """True if the string contains at least one character in the Arabic Unicode block."""
    for c in s:
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F":
            return True
    return False


def _detect_language(text: str) -> Literal["ar", "en"]:
    """Auto-detect language from script: Arabic script -> ar, else en."""
    if _has_arabic_chars(text):
        return "ar"
    return "en"


def synthesize(
    text: str,
    language: Literal["auto", "ar", "en"] = "auto",
    speaker: Optional[str] = None,
) -> tuple[bytes, int]:
    """
    Convert text to speech and return WAV bytes and sample rate.

    Args:
        text: Text to synthesize (Arabic).
        language: "auto" (detect from script), "ar" (Arabic), or "en" (English).
                  Note: This model primarily supports Arabic. English requests will be treated as Arabic.
        speaker: Not used (kept for API compatibility).

    Returns:
        Tuple of (wav_bytes, sample_rate).
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    t = text.strip()
    lang = language
    if lang == "auto":
        lang = _detect_language(t)
    
    # NAMAA-Saudi-TTS is optimized for Arabic, but we'll use it for any text
    # If language is explicitly "en", we still process it (model may handle it)
    if lang not in ("ar", "en"):
        lang = "ar"
    
    model = _get_tts()

    # Generate speech
    with torch.no_grad():
        wav = model.generate(t, language_id="ar")
    
    # Convert to bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        ta.save(tmp_path, wav.cpu(), model.sr)
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes, model.sr
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
