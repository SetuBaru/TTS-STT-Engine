"""
Multilingual Text-to-Speech using Coqui XTTS-v2.
Supports Arabic and English (and other XTTS-v2 languages) with a single model.
"""

import io
from typing import Optional, Literal

# --- Transformers compatibility: Coqui TTS does "from transformers import BeamSearchScorer".
#     Newer transformers use a LazyModule that only exposes names in _class_to_module;
#     BeamSearchScorer exists in generation.beam_search but isn't in that map. Register it.
_transformers_patched = False

def _patch_transformers_for_tts():
    global _transformers_patched
    if _transformers_patched:
        return
    try:
        import transformers  # noqa: F401
        # LazyModule resolves names via _class_to_module; add BeamSearchScorer -> generation.beam_search
        if hasattr(transformers, "_class_to_module"):
            transformers._class_to_module["BeamSearchScorer"] = "generation.beam_search"
        if hasattr(transformers, "_objects"):
            from transformers.generation.beam_search import BeamSearchScorer
            transformers._objects["BeamSearchScorer"] = BeamSearchScorer
        if hasattr(transformers, "__all__") and "BeamSearchScorer" not in transformers.__all__:
            transformers.__all__.append("BeamSearchScorer")
        _transformers_patched = True
    except Exception:
        pass

_patch_transformers_for_tts()

# Lazy-loaded model
_tts = None

# XTTS-v2 sample rate (fixed)
SAMPLE_RATE = 24000

# Default preset speaker (works for both Arabic and English in XTTS-v2)
DEFAULT_SPEAKER = "Claribel Dervla"


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


def _get_tts():
    global _tts
    if _tts is None:
        # Ensure patch is applied (in case transformers was loaded after our module)
        _patch_transformers_for_tts()
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts


def synthesize(
    text: str,
    language: Literal["auto", "ar", "en"] = "auto",
    speaker: Optional[str] = None,
) -> tuple[bytes, int]:
    """
    Convert text to speech and return WAV bytes and sample rate.

    Args:
        text: Text to synthesize (Arabic or English).
        language: "auto" (detect from script), "ar" (Arabic), or "en" (English).
        speaker: Preset speaker name; uses DEFAULT_SPEAKER if not set.

    Returns:
        Tuple of (wav_bytes, sample_rate).
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    t = text.strip()
    lang = language
    if lang == "auto":
        lang = _detect_language(t)
    if lang not in ("ar", "en"):
        lang = "en"

    spk = speaker or DEFAULT_SPEAKER
    model = _get_tts()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.tts_to_file(
            text=t,
            speaker=spk,
            language=lang,
            file_path=tmp_path,
        )
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes, SAMPLE_RATE
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
