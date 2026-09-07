import os
import random
import numpy as np
import soundfile as sf

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
        clean_files = noisy_files = file_names

    segment_len = segment_seconds * sample_rate

    total_segments = 0

    for clean_file, noisy_file in zip(clean_files, noisy_files):

        clean_audio, _ = sf.read(os.path.join(input_clean_dir, clean_file), dtype="float32")
        noisy_audio, _ = sf.read(os.path.join(input_noisy_dir, noisy_file), dtype="float32")

        #streo -> mono
        if clean_audio.ndim == 2: clean_audio = clean_audio.mean(axis=1)
        if noisy_audio.ndim == 2: noisy_audio = noisy_audio.mean(axis=1)

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
    # Split by whole recording BEFORE slicing into segments, so segments from
    # the same recording never land on both sides — that would leak info
    # between train and validation (neighboring segments look almost identical)
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