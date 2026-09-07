import tensorflow as tf
import numpy as np

from model.unet import build_unet
from audio.stft_utils import wav_to_mag_phase, mag_phase_to_wav


def load_model(weights_path):

    model = build_unet()

    dummy_input = tf.zeros((1, 497, 257, 1))
    model(dummy_input, training=False)

    model.load_weights(weights_path)

    return model


def enhance_audio(model, noisy_wav):

    original_length = len(noisy_wav)

    # [T] → [1, T]  # batch
    noisy_tensor = tf.constant(noisy_wav[np.newaxis, :], dtype=tf.float32)

    magnitude, phase = wav_to_mag_phase(noisy_tensor)

    # [1, T, F] → [1, T, F, 1]  # channel
    magnitude = tf.expand_dims(magnitude, axis=-1)

    enhanced_magnitude = model(magnitude, training=False)

    # [1, T, F, 1] → [1, T, F]  # remove channel
    enhanced_magnitude = tf.squeeze(
        enhanced_magnitude,
        axis=-1
    )

    enhanced_wav = mag_phase_to_wav(
        enhanced_magnitude,
        phase,
        target_len=original_length
    )

    # [1, T] → [T]
    return enhanced_wav[0].numpy()
