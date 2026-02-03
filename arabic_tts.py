"""
Arabic Text-to-Speech using Meta's MMS-TTS (facebook/mms-tts-ara).
Generates audio from Arabic text without requiring a voice prompt.
"""

import io
from typing import Optional

# Lazy-loaded pipeline to avoid loading at import time
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        _pipeline = pipeline(
            task="text-to-speech",
            model="facebook/mms-tts-ara",
            device=device,
        )
    return _pipeline


def _has_arabic_chars(s: str) -> bool:
    """True if the string contains at least one character in the Arabic Unicode block."""
    for c in s:
        if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F":
            return True
    return False


def synthesize(text: str, seed: Optional[int] = 42) -> tuple[bytes, int]:
    """
    Convert Arabic text to speech and return WAV bytes and sample rate.

    Args:
        text: Arabic text to synthesize.
        seed: Random seed for reproducibility (VITS is non-deterministic).

    Returns:
        Tuple of (wav_bytes, sample_rate).
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    t = text.strip()
    if not _has_arabic_chars(t):
        raise ValueError(
            "This model expects Arabic text only. Please enter text in Arabic (e.g. مرحبا، كيف حالك؟). "
            "English or other scripts are not supported."
        )

    from scipy.io.wavfile import write as wav_write

    pipe = _get_pipeline()
    if seed is not None:
        from transformers import set_seed
        set_seed(seed)

    try:
        out = pipe(t)
    except Exception as e:
        err_msg = str(e).lower()
        if "size 0" in err_msg or "negative" in err_msg or "dimension" in err_msg:
            raise ValueError(
                "The model could not process this text. Please use Arabic text only."
            ) from e
        raise

    audio = out["audio"]
    sample_rate = out["sampling_rate"]

    buf = io.BytesIO()
    wav_write(buf, sample_rate, audio.squeeze())
    buf.seek(0)
    return buf.read(), sample_rate
