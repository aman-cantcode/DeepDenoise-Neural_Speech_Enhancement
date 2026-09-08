import tensorflow as tf
import numpy as np
import h5py

from model.unet import build_unet
from audio.stft_utils import wav_to_mag_phase, mag_phase_to_wav


def _load_named_weights(model, weights_path):
    conv_layers = [
        layer for layer in model.layers
        if layer.__class__.__name__ == "Conv2D"
    ]
    batch_norm_layers = [
        layer for layer in model.layers
        if layer.__class__.__name__ == "BatchNormalization"
    ]
    transpose_layers = [
        layer for layer in model.layers
        if layer.__class__.__name__ == "Conv2DTranspose"
    ]

    block_names = [
        "encoder_block1", "encoder_block2", "encoder_block3",
        "encoder_block4", "bottleneck", "decoder_block4",
        "decoder_block3", "decoder_block2", "decoder_block1"
    ]

    with h5py.File(weights_path, "r") as weights_file:
        conv_index = 0
        batch_norm_index = 0
        for block_name in block_names:
            block = weights_file[block_name]
            for conv_name, batch_norm_name in (
                ("conv1", "bn1"),
                ("conv2", "bn2")
            ):
                conv = block[conv_name]["vars"]
                batch_norm = block[batch_norm_name]["vars"]
                conv_layers[conv_index].set_weights([
                    conv["0"][:],
                    conv["1"][:]
                ])
                batch_norm_layers[batch_norm_index].set_weights([
                    batch_norm[str(index)][:] for index in range(4)
                ])
                conv_index += 1
                batch_norm_index += 1

        for index, layer in enumerate(transpose_layers):
            group = weights_file["layers"][
                "conv2d_transpose" if index == 0
                else f"conv2d_transpose_{index}"
            ]["vars"]
            layer.set_weights([group["0"][:], group["1"][:]])

        output = weights_file["layers"]["conv2d"]["vars"]
        conv_layers[-1].set_weights([output["0"][:], output["1"][:]])


def load_model(weights_path):

    model = build_unet()

    dummy_input = tf.zeros((1, 257, 497, 1))
    model(dummy_input, training=False)

    load_weights(model, weights_path)

    return model


def load_weights(model, weights_path):

    with h5py.File(weights_path, "r") as weights_file:
        uses_named_blocks = "encoder_block1" in weights_file

    if uses_named_blocks:
        _load_named_weights(model, weights_path)
    else:
        model.load_weights(weights_path)


def enhance_audio(model, noisy_wav):

    noisy_wav = np.asarray(noisy_wav, dtype=np.float32)
    if noisy_wav.ndim != 1 or noisy_wav.size == 0:
        raise ValueError("noisy_wav must be a non-empty mono waveform")
    if not np.isfinite(noisy_wav).all():
        raise ValueError("noisy_wav must contain only finite values")

    original_length = len(noisy_wav)

    # [T] → [1, T]  # batch
    noisy_tensor = tf.constant(noisy_wav[np.newaxis, :], dtype=tf.float32)

    magnitude, phase = wav_to_mag_phase(noisy_tensor)

    # [1, F, T] → [1, F, T, 1]  # channel
    magnitude = tf.expand_dims(magnitude, axis=-1)

    enhanced_magnitude = model(magnitude, training=False)

    # [1, F, T, 1] → [1, F, T]  # remove channel
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
