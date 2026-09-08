import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from audio.audio_utils import load_audio
from audio.enhance_utils import load_model, enhance_audio
from evaluation.metrics import calculate_metrics


WEIGHTS_PATH = str(BASE_DIR / "weights/unet_tf_weights.weights.h5")
CLEAN_DIR = str(BASE_DIR / "dataset/test/clean")
NOISY_DIR = str(BASE_DIR / "dataset/test/noisy")

OUTPUT_DIR = str(BASE_DIR / "dataset/test/outputs")
ENHANCED_DIR = os.path.join(OUTPUT_DIR, "enhanced")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")

SAMPLE_RATE = 16000


def plot_average_metrics(noisy_scores, enhanced_scores):

    metrics = ["STOI", "PESQ", "SNR"]

    noisy_average = []
    enhanced_average = []

    for metric in metrics:
        noisy_average.append(np.mean(noisy_scores[metric]))
        enhanced_average.append(np.mean(enhanced_scores[metric]))

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        noisy_average,
        width,
        label="Noisy"
    )

    plt.bar(
        x + width / 2,
        enhanced_average,
        width,
        label="Enhanced"
    )

    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Average Metrics: Noisy vs Enhanced")
    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        CHARTS_DIR,
        "average_metrics.png"
    )

    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def plot_improvement(noisy_scores, enhanced_scores, metric):

    noisy = np.array(noisy_scores[metric])
    enhanced = np.array(enhanced_scores[metric])

    improvement = enhanced - noisy

    plt.figure(figsize=(10, 4))

    plt.plot(
        improvement,
        marker="o"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("File index")
    plt.ylabel(f"Improvement in {metric}")
    plt.title(f"{metric} Improvement Per File")

    plt.tight_layout()

    path = os.path.join(
        CHARTS_DIR,
        f"improvement_{metric.lower()}.png"
    )

    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")



def evaluate():

    os.makedirs(ENHANCED_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        print("Weights file not found:", WEIGHTS_PATH)
        return

    if not os.path.isdir(CLEAN_DIR) or not os.path.isdir(NOISY_DIR):
        raise FileNotFoundError(
            f"Test directories not found: {CLEAN_DIR}, {NOISY_DIR}"
        )

    print("\nStarting evaluation...\n")

    model = load_model(WEIGHTS_PATH)

    clean_files = sorted(
        f for f in os.listdir(CLEAN_DIR)
        if f.endswith(".wav")
    )

    noisy_files = sorted(
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    )

    files = [ 
        f for f in clean_files
        if f in noisy_files
    ]

    if not files:
        raise ValueError("No matching .wav files found in the test directories")

    print(f"Number of test files: {len(files)}\n")

    noisy_scores = {
        "STOI": [],
        "PESQ": [],
        "SNR": []
    }

    enhanced_scores = {
        "STOI": [],
        "PESQ": [],
        "SNR": []
    }

    for i, filename in enumerate(files):

        clean_path = os.path.join(CLEAN_DIR, filename)

        noisy_path = os.path.join(NOISY_DIR, filename)

        clean, _ = load_audio(clean_path, expected_sample_rate=SAMPLE_RATE)
        noisy, _ = load_audio(noisy_path, expected_sample_rate=SAMPLE_RATE)

        enhanced = enhance_audio(
            model,
            noisy
        )

        output_path = os.path.join(
            ENHANCED_DIR,
            filename
        )

        sf.write(
            output_path,
            enhanced,
            SAMPLE_RATE
        )

        length = min(len(clean), len(noisy), len(enhanced))

        clean = clean[:length]
        noisy = noisy[:length]
        enhanced = enhanced[:length]

        stoi_noisy, pesq_noisy, snr_noisy = calculate_metrics(
            clean,
            noisy,
            SAMPLE_RATE
        )

        stoi_enhanced, pesq_enhanced, snr_enhanced = calculate_metrics(
            clean,
            enhanced,
            SAMPLE_RATE
        )

        noisy_scores["STOI"].append(stoi_noisy)
        noisy_scores["PESQ"].append(pesq_noisy)
        noisy_scores["SNR"].append(snr_noisy)

        enhanced_scores["STOI"].append(stoi_enhanced)
        enhanced_scores["PESQ"].append(pesq_enhanced)
        enhanced_scores["SNR"].append(snr_enhanced)

        print(f"[{i + 1}/{len(files)}] {filename}")

        print(
            f"  Noisy     : "
            f"STOI={stoi_noisy:.4f}, "
            f"PESQ={pesq_noisy:.4f}, "
            f"SNR={snr_noisy:.2f} dB"
        )

        print(
            f"  Enhanced  : "
            f"STOI={stoi_enhanced:.4f}, "
            f"PESQ={pesq_enhanced:.4f}, "
            f"SNR={snr_enhanced:.2f} dB\n"
        )

    print("=" * 60)
    print("AVERAGE RESULTS")
    print("=" * 60)

    report = []

    for metric in ["STOI", "PESQ", "SNR"]:

        noisy_average = np.mean(noisy_scores[metric])

        enhanced_average = np.mean(enhanced_scores[metric])

        improvement = (enhanced_average - noisy_average)

        print(
            f"{metric}: "
            f"{noisy_average:.4f} -> "
            f"{enhanced_average:.4f} "
            f"(Improvement: {improvement:+.4f})"
        )

        report.append(
            f"{metric}: "
            f"Noisy={noisy_average:.4f}, "
            f"Enhanced={enhanced_average:.4f}, "
            f"Improvement={improvement:+.4f}\n"
        )


    report_dir = os.path.join(
        OUTPUT_DIR,
        "reports"
    )

    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(
        report_dir,
        "metrics_report.txt"
    )

    with open(report_path, "w") as file:
        file.writelines(report)

    print(f"\nReport saved: {report_path}")


    print("\nCreating charts...\n")

    plot_average_metrics(
        noisy_scores,
        enhanced_scores
    )

    for metric in ["STOI", "PESQ", "SNR"]:
        plot_improvement(
            noisy_scores,
            enhanced_scores,
            metric
        )

    print("\nEvaluation complete!")

if __name__ == "__main__":
    evaluate()