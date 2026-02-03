"""
Apply transformers compatibility patch for Coqui TTS (BeamSearchScorer) on import.
Must be imported before any code that loads TTS. api.py imports this first.
"""
import sys
import types


def _apply():
    import transformers as _real
    from transformers.generation.beam_search import BeamSearchScorer

    class _TransformersModule(types.ModuleType):
        """Wrapper so 'from transformers import BeamSearchScorer' works."""
        __name__ = "transformers"
        BeamSearchScorer = None  # set below

        def __init__(self, real):
            super().__init__(real.__name__)
            self.__dict__.update(real.__dict__)
            self._real = real
            self.BeamSearchScorer = BeamSearchScorer

        def __getattr__(self, name):
            if name in ("_real", "BeamSearchScorer"):
                return object.__getattribute__(self, name)
            return getattr(self._real, name)

    wrapper = _TransformersModule(_real)
    if hasattr(wrapper, "__all__") and "BeamSearchScorer" not in wrapper.__all__:
        wrapper.__all__ = list(wrapper.__all__) + ["BeamSearchScorer"]
    sys.modules["transformers"] = wrapper


_apply()
