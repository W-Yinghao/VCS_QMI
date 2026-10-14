"""Scoped import stubs for third-party code paths V7X does not execute (recorded deviations; no third-party file is edited):
- fairseq (fallback only, when it is not installed): ModelScope 1.20.1's multi_modal preprocessor module loads its OFA package (speech, ViT,
  generation code importing fairseq) alongside mPLUG's preprocessor; the mPLUG code imports nothing from fairseq.  TIFA's requirements.txt lists
  fairseq, so the env installs fairseq 0.12.2 when it builds; the stub exists only for a failed build.
- promptcap, lavis.models: imported at module level by TIFA's tifascore/vqa_models.py for its PromptCap / BLIP-2 routes; the mplug-large route
  never calls them.
Every stub raises if it is ever called."""
import sys
import types


def _unavailable(*a, **k):
    raise RuntimeError("V7X scoped-import stub: this third-party route is not used")


class _Stub:
    def __init__(self, *a, **k):
        _unavailable()

    @classmethod
    def from_config(cls, *a, **k):
        _unavailable()


class _FairseqFinder:
    """Fallback only when fairseq is not installed: every fairseq.* import resolves to a stub package whose attributes are stub classes / decorators
    (identity when used as a decorator, raising when instantiated).  The mPLUG code imports nothing from fairseq (only ModelScope's OFA package does)."""

    def find_spec(self, name, path=None, target=None):
        if name == "fairseq" or name.startswith("fairseq."):
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name); m.__path__ = []; m.__getattr__ = _fairseq_attr; return m

    def exec_module(self, module):
        return None


def _fairseq_attr(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return type(name, (_Stub,), {"__init_subclass__": classmethod(lambda cls, **k: None)}) if name[:1].isupper() else _decorator_or_stub


def _decorator_or_stub(*a, **k):
    if len(a) == 1 and not k and callable(a[0]):
        return a[0]                                                            # used as a decorator at import time: identity
    _unavailable()


def install(tifa=False):
    import importlib.util
    if importlib.util.find_spec("fairseq") is None and not any(isinstance(f, _FairseqFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FairseqFinder())
    if tifa:
        pc = types.ModuleType("promptcap"); pc.PromptCap_VQA = _Stub; sys.modules.setdefault("promptcap", pc)
        lv = types.ModuleType("lavis"); lvm = types.ModuleType("lavis.models"); lvm.load_model_and_preprocess = _unavailable; lv.models = lvm
        sys.modules.setdefault("lavis", lv); sys.modules.setdefault("lavis.models", lvm)
