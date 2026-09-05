import os
import glob

import numpy as np
import soundfile as sf
import tensorflow as tf

MAX_AUDIO_LEN = 64000


def _fix_length(audio, target_len): # _func nomenclature?

    if len(audio) < target_len: audio = np.pad(audio, (0, target_len - len(audio)))
    else: audio = audio[:target_len]
    
    return audio.astype(np.float32)


def _load_pair(noisy_path, clean_path, max_len):
    #paths here coming from TensorFlow(bytes objects) rather than normal strings
    noisy, _ = sf.read(noisy_path.decode())
    clean, _ = sf.read(clean_path.decode())

    noisy = _fix_length(noisy, max_len)
    clean = _fix_length(clean, max_len)

    return noisy, clean


def make_dataset(noisy_dir, clean_dir, batch_size=4, max_len=MAX_AUDIO_LEN, shuffle=True):
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))

    if len(noisy_files) == 0:
        raise FileNotFoundError(f"No .wav files found in: {noisy_dir}")
    if len(noisy_files) != len(clean_files):
        raise ValueError(
            f"Noisy ({len(noisy_files)}) and clean ({len(clean_files)}) "
            "directories must have the same number of files."
        )

    print(f"  Found {len(noisy_files)} audio pairs in dataset.")

    noisy_paths = tf.constant(noisy_files) #TensorFlow string tensor of file paths
    clean_paths = tf.constant(clean_files)
    

    path_dataset = tf.data.Dataset.from_tensor_slices((noisy_paths, clean_paths))

    if shuffle:
        path_dataset = path_dataset.shuffle(buffer_size=len(noisy_files), reshuffle_each_iteration=True)

    def load_pair_tf(noisy_path, clean_path):
        noisy, clean = tf.numpy_function(
            func=lambda n, c: _load_pair(n, c, max_len),
            inp=[noisy_path, clean_path],
            Tout=[tf.float32, tf.float32]
        )
        noisy.set_shape([max_len])
        clean.set_shape([max_len])
        return noisy, clean

    dataset = (
        path_dataset
        .map(load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset
