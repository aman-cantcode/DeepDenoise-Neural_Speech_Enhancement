import os
import numpy as np
import soundfile as sf


def slice_audio(
    input_clean_dir,
    input_noisy_dir,
    output_clean_dir,
    output_noisy_dir,
    segment_seconds=4,
    sample_rate=16000,
):
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    clean_files = sorted(
        f for f in os.listdir(input_clean_dir)
        if f.endswith(".wav")
    )

    noisy_files = sorted(
        f for f in os.listdir(input_noisy_dir)
        if f.endswith(".wav")
    )

    segment_len = segment_seconds * sample_rate

    total_segments = 0

    for clean_file, noisy_file in zip(clean_files, noisy_files):

        clean_audio, _ = sf.read(
            os.path.join(input_clean_dir, clean_file),
            dtype="float32"
        )

        noisy_audio, _ = sf.read(
            os.path.join(input_noisy_dir, noisy_file),
            dtype="float32"
        )

        if clean_audio.ndim == 2: clean_audio = clean_audio.mean(axis=1)
        if noisy_audio.ndim == 2: noisy_audio = noisy_audio.mean(axis=1)

        # tiny length mismatches
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


if __name__ == "__main__":
    slice_audio(
        input_clean_dir="dataset/raw/clean",
        input_noisy_dir="dataset/raw/noisy",
        output_clean_dir="dataset/train/clean",
        output_noisy_dir="dataset/train/noisy",
    )