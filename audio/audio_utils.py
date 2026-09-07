import numpy as np
import soundfile as sf


def load_audio(path):

    audio, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    #streo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sample_rate


def fix_length(audio, target_len):

    if len(audio) < target_len:
        audio = np.pad(
            audio,
            (0, target_len - len(audio))
        )
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)
