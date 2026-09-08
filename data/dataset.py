import os
import glob

import tensorflow as tf

from audio.audio_utils import load_audio, fix_length

MAX_AUDIO_LEN = 64000



def _load_pair(noisy_path, clean_path, max_len): # _func nomenclature?

    #paths here coming from TensorFlow(bytes objects) rather than normal strings
    noisy, _ = load_audio(noisy_path.decode())
    clean, _ = load_audio(clean_path.decode())

    noisy = fix_length(noisy, max_len)
    clean = fix_length(clean, max_len)

    return noisy, clean


def make_dataset(noisy_dir, clean_dir, batch_size=4, max_len=MAX_AUDIO_LEN, shuffle=True):
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))

    if len(noisy_files) == 0:
        raise FileNotFoundError(f"No .wav files found in: {noisy_dir}")
    noisy_by_name = {os.path.basename(path): path for path in noisy_files}
    clean_by_name = {os.path.basename(path): path for path in clean_files}
    if set(noisy_by_name) != set(clean_by_name):
        missing_noisy = sorted(set(clean_by_name) - set(noisy_by_name))
        missing_clean = sorted(set(noisy_by_name) - set(clean_by_name))
        raise ValueError(
            "Noisy and clean directories must contain matching filenames. "
            f"Missing noisy: {missing_noisy}; missing clean: {missing_clean}"
        )

    noisy_files = [noisy_by_name[name] for name in sorted(noisy_by_name)]
    clean_files = [clean_by_name[name] for name in sorted(clean_by_name)]

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
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset


