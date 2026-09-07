import os
import sys
import argparse

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.unet import build_unet
from audio.stft_utils import wav_to_mag_phase, mag_phase_to_wav
from evaluation.metrics import calculate_metrics


WEIGHTS_PATH = "weights/unet_tf_weights.weights.h5"
SAMPLE_RATE = 16000


def load_model(weights_path):

    model = build_unet()

    dummy = tf.zeros([1, 497, 257, 1], dtype=tf.float32)
    model(dummy, training=False)

    model.load_weights(weights_path)

    print(f"  Model loaded from: {weights_path}")

    return model


def enhance(model, noisy_wav):

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


def get_spectrogram(audio):

    magnitude, _ = wav_to_mag_phase(tf.expand_dims(audio, axis=0))

    # [1, T, F] → [T, F]
    magnitude = magnitude[0]

    # db = 20 * log10(magnitude)
    magnitude_db = (20 * tf.math.log(magnitude + 1e-8)/tf.math.log(10.0)) #1e-8 to avoid log(0)

    return magnitude_db.numpy()


def plot_spectrogram(audio, title, axis):

    spectrogram = get_spectrogram(audio)

    image = axis.imshow(
        spectrogram.T, #transpose => x : time, y : frequency
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest"
    )

    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Time frames")
    axis.set_ylabel("Frequency bins")

    plt.colorbar(
        image,
        ax=axis,
        label="dB"
    )


def plot_results(
    noisy,
    enhanced,
    output_folder,
    clean=None,
    sample_rate=SAMPLE_RATE
):

    # Waveform comparison
    figure, axis = plt.subplots(figsize=(12, 4))

    time = np.arange(len(noisy)) / sample_rate

    axis.plot(
        time,   # x
        noisy,  # y : noisy amplitude
        label="Noisy",
        alpha=0.7,
        linewidth=0.8
    )

    axis.plot(
        time,
        enhanced,
        label="Enhanced",
        alpha=0.7,
        linewidth=0.8
    )

    if clean is not None:
        axis.plot(
            time,
            clean,
            label="Clean",
            alpha=0.7,
            linewidth=0.8
        )

    axis.set_title("Waveform Comparison", fontsize=14)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "waveform_comparison.png"
        ),
        dpi=150
    )

    plt.close()


    # Spectrogram comparison
    if clean is not None:

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4)
        )

        plot_spectrogram(noisy, "Noisy", axes[0])
        plot_spectrogram(enhanced, "Enhanced", axes[1])
        plot_spectrogram(clean, "Clean", axes[2])

    else:

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4)
        )

        plot_spectrogram(noisy, "Noisy", axes[0])
        plot_spectrogram(enhanced, "Enhanced", axes[1])
        

    plt.suptitle(
        "Spectrogram Comparison",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "spectrogram_comparison.png"
        ),
        dpi=150
    )

    plt.close()


    # Difference spectrogram
    noisy_spectrogram = get_spectrogram(noisy)
    enhanced_spectrogram = get_spectrogram(enhanced)

    min_time = min(noisy_spectrogram.shape[0], enhanced_spectrogram.shape[0])

    difference = (enhanced_spectrogram[:min_time, :] - noisy_spectrogram[:min_time, :])

    figure, axis = plt.subplots(
        figsize=(10, 4)
    )

    image = axis.imshow(
        difference.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm"
    )

    plt.colorbar(
        image,
        ax=axis,
        label="dB difference"
    )

    axis.set_title(
        "Difference Spectrogram: Enhanced − Noisy\n"
        "(blue = noise removed, red = boosted)",
        fontsize=12
    )

    axis.set_xlabel("Time frames")
    axis.set_ylabel("Frequency bins")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "difference_spectrogram.png"
        ),
        dpi=150
    )

    plt.close()

    print(f"  Plots saved to: {output_folder}")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Enhance a noisy audio file."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to noisy input .wav file"
    )

    parser.add_argument(
        "--clean",
        default=None,
        help="Optional clean reference .wav file"
    )

    parser.add_argument(
        "--weights",
        default=WEIGHTS_PATH,
        help="Path to .h5 weights file"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 60)
    print("  Speech Enhancement Inference")
    print("=" * 60)

    if not os.path.exists(args.weights):
        sys.exit(
            f"Weights not found: {args.weights}\n"
            "Train the model first."
        )

    model = load_model(args.weights)

    noisy, sample_rate = sf.read(args.input)

    if noisy.ndim > 1: noisy = noisy.mean(axis=1)
    noisy = noisy.astype(np.float32)

    print(
        f"  Input: {args.input} "
        f"({len(noisy) / sample_rate:.1f}s, "
        f"{sample_rate}Hz)"
    )

    print("  Enhancing...")

    enhanced = enhance(
        model,
        noisy
    )

    file_name = os.path.basename(args.input)
    file_name_without_extension = os.path.splitext(file_name)[0]

    results_dir = os.path.join(
        "samples",
        "outputs",
        file_name_without_extension
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        results_dir,
        "enhanced.wav"
    )

    sf.write(
        output_file,
        enhanced,
        sample_rate
    )

    print(f"  Enhanced audio saved to: {output_file}")

    clean = None

    if args.clean:

        clean, clean_sample_rate = sf.read(args.clean)

        if clean.ndim > 1: clean = clean.mean(axis=1)
        clean = clean.astype(np.float32)

        min_length = min(len(clean), len(enhanced))

        stoi_score, pesq_score, snr_score = calculate_metrics(
            clean[:min_length],
            enhanced[:min_length],
            sample_rate
        )

        print()
        print("  Quality Metrics")
        print(f"  STOI : {stoi_score:.4f}")
        print(f"  PESQ : {pesq_score:.4f}")
        print(f"  SNR  : {snr_score:.2f} dB")


    print()
    print("  Generating plots...")

    min_length = min(len(noisy), len(enhanced))

    noisy = noisy[:min_length]
    enhanced = enhanced[:min_length]

    if clean is not None:
        clean = clean[:min_length]

    plot_results(
        noisy,
        enhanced,
        results_dir,
        clean,
        sample_rate
    )

    print()
    print("  Done.")
    print(f"  All results: {results_dir}/")


if __name__ == "__main__":
    main()