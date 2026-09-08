import tensorflow as tf

N_FFT      = 512   # FFT size → freq_bins = N_FFT // 2 + 1 = 257
HOP_LENGTH = 128  
WIN_LENGTH = 512 


def wav_to_mag_phase(waveform, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH):

    # waveform: float32 tensor [B, T], where B = batch size, T = number of audio samples(4 sec)
    if len(waveform.shape) == 1: waveform = tf.expand_dims(waveform, axis=0)

    stft_output = tf.signal.stft(
        waveform,
        frame_length=win,
        frame_step=hop,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window
    ) #shape: [B, time_frames, freq_bins]

    stft_output = tf.transpose(stft_output, perm=[0, 2, 1])  # [B, time, freq] -> [B, freq, time]


    mag   = tf.abs(stft_output)
    phase = tf.math.angle(stft_output)

    return mag, phase


def mag_phase_to_wav(mag, phase, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH, target_len=None):

    # U-Net decder can produce a magnitude that is 1–2 bins smaller than the original phase due to _match_size crops on odd dimensions.
    min_freq = tf.minimum(tf.shape(mag)[1], tf.shape(phase)[1])
    min_time = tf.minimum(tf.shape(mag)[2], tf.shape(phase)[2])

    mag   = mag[:, :min_freq, :min_time]
    phase = phase[:, :min_freq, :min_time]
    
    real = mag * tf.cos(phase)
    imag = mag * tf.sin(phase)
    stft_complex = tf.complex(real, imag)

    stft_complex = tf.transpose(stft_complex, perm=[0, 2, 1])

    inverse_window_fn = tf.signal.inverse_stft_window_fn(
        frame_step=hop,
        forward_window_fn=tf.signal.hann_window
    )

    wav = tf.signal.inverse_stft(
        stfts=stft_complex,
        frame_length=win,
        frame_step=hop,
        fft_length=n_fft,
        window_fn=inverse_window_fn
    )

    if target_len is not None:
        current_len = tf.shape(wav)[1]
        padding = tf.maximum(target_len - current_len, 0)
        wav = tf.pad(wav, [[0, 0], [0, padding]])
        wav = wav[:, :target_len]

    return wav
