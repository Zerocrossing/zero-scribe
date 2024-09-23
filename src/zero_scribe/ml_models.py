import whisperx
import gc
from zero_scribe.consts import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE
import torch

model = None


def load_whisper_model():
    global model
    if model is None:
        model = whisperx.load_model(
            "large-v2", WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return model


def unload_whisper_model():
    global model
    if model is not None:
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
