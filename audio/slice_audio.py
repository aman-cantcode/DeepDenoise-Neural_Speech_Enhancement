import os
import random
import soundfile as sf
import numpy as np

from audio.audio_utils import load_audio

VALID_SPLIT = 0.1
SEED        = 42


def slice_audio(
    input_clean_dir,
    input_noisy_dir,
    output_clean_dir,
    output_noisy_dir,
    file_names=None, # list of files to slice for test set
    segment_seconds=4,
    sample_rate=16000,
):
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    if segment_seconds <= 0 or sample_rate <= 0:
        raise ValueError("segment_seconds and sample_rate must be positive")

    if file_names is None:
        clean_files = sorted(
            f for f in os.listdir(input_clean_dir)
            if f.endswith(".wav")
        )

        noisy_files = sorted(
            f for f in os.listdir(input_noisy_dir)
            if f.endswith(".wav")
        )
    else:
        clean_files = list(file_names)
        noisy_files = list(file_names)

        missing_clean = [
            name for name in clean_files
            if not os.path.isfile(os.path.join(input_clean_dir, name))
        ]
        missing_noisy = [
            name for name in noisy_files
            if not os.path.isfile(os.path.join(input_noisy_dir, name))
        ]
        if missing_clean or missing_noisy:
            raise FileNotFoundError(
                f"Missing clean files: {missing_clean}; "
                f"missing noisy files: {missing_noisy}"
            )

    clean_names = set(clean_files)
    noisy_names = set(noisy_files)
    if clean_names != noisy_names:
        raise ValueError("Clean and noisy recordings must have matching filenames")

    segment_len = segment_seconds * sample_rate

    total_segments = 0

    for clean_file in clean_files:
        noisy_file = clean_file

        clean_audio, _ = load_audio(os.path.join(input_clean_dir, clean_file))
        noisy_audio, _ = load_audio(os.path.join(input_noisy_dir, noisy_file))

        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        stem = os.path.splitext(clean_file)[0]

        for start in range(0, min_len, segment_len):

            clean_segment = clean_audio[start : start + segment_len]
            noisy_segment = noisy_audio[start : start + segment_len]

            if len(clean_segment) < segment_len:
                pad = segment_len - len(clean_segment)

                clean_segment = np.pad(clean_segment, (0, pad))
                noisy_segment = np.pad(noisy_segment, (0, pad))

            
            name = f"{stem}_{start}.wav"

            sf.write(
                os.path.join(output_clean_dir, name),
                clean_segment,
                sample_rate
            )

            sf.write(
                os.path.join(output_noisy_dir, name),
                noisy_segment,
                sample_rate
            )

            total_segments += 1

    print(f"Done. Saved {total_segments} segments.")


def split_and_slice(
    input_clean_dir,
    input_noisy_dir,
    train_clean_dir,
    train_noisy_dir,
    valid_clean_dir,
    valid_noisy_dir,
    valid_split=VALID_SPLIT,
    segment_seconds=4,
    sample_rate=16000,
):
    
    clean_files = sorted(
        f for f in os.listdir(input_clean_dir)
        if f.endswith(".wav")
    )

    shuffled = clean_files.copy()
    random.seed(SEED)
    random.shuffle(shuffled)

    num_valid   = max(1, int(len(shuffled) * valid_split))
    valid_files = shuffled[:num_valid]
    train_files = shuffled[num_valid:]

    print(f"Splitting {len(clean_files)} recordings → {len(train_files)} train, {len(valid_files)} valid")

    print("\nSlicing train set...")
    slice_audio(
        input_clean_dir, input_noisy_dir,
        train_clean_dir, train_noisy_dir,
        file_names=train_files,
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
    )

    print("\nSlicing valid set...")
    slice_audio(
        input_clean_dir, input_noisy_dir,
        valid_clean_dir, valid_noisy_dir,
        file_names=valid_files,
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
    )


if __name__ == "__main__":
    split_and_slice(
        input_clean_dir="dataset/raw/clean",
        input_noisy_dir="dataset/raw/noisy",
        train_clean_dir="dataset/train/clean",
        train_noisy_dir="dataset/train/noisy",
        valid_clean_dir="dataset/valid/clean",
        valid_noisy_dir="dataset/valid/noisy",
    )