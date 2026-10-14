"""Scoped import stubs for third-party code paths V7X does not execute (recorded deviations; no third-party file is edited):
- fairseq: imported at module level by ModelScope's OFA speech preprocessor (modelscope/preprocessors/ofa/asr.py: CompositeAudioFeatureTransform,
  S2TDataConfig), which ModelScope's multi_modal preprocessor module loads alongside mPLUG's; the mPLUG VQA route never calls it.
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


def install(tifa=False):
    for name in ("fairseq", "fairseq.data", "fairseq.data.audio", "fairseq.data.audio.feature_transforms", "fairseq.data.audio.speech_to_text_dataset"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["fairseq.data.audio.feature_transforms"].CompositeAudioFeatureTransform = _Stub
    sys.modules["fairseq.data.audio.speech_to_text_dataset"].S2TDataConfig = _Stub
    if tifa:
        pc = types.ModuleType("promptcap"); pc.PromptCap_VQA = _Stub; sys.modules.setdefault("promptcap", pc)
        lv = types.ModuleType("lavis"); lvm = types.ModuleType("lavis.models"); lvm.load_model_and_preprocess = _unavailable; lv.models = lvm
        sys.modules.setdefault("lavis", lv); sys.modules.setdefault("lavis.models", lvm)
