import numpy as np
from pystoi import stoi
from pesq import pesq


def calculate_stoi(clean, enhanced, sample_rate=16000):
    min_len = min(len(clean), len(enhanced))
    if min_len == 0:
        raise ValueError("Signals must be non-empty")
    return stoi(clean[:min_len], enhanced[:min_len], sample_rate, extended=False)


def calculate_pesq(clean, enhanced, sample_rate=16000):
    min_len = min(len(clean), len(enhanced))
    if min_len == 0:
        raise ValueError("Signals must be non-empty")
    return pesq(sample_rate, clean[:min_len], enhanced[:min_len], "wb")


def calculate_snr(clean, enhanced):
    min_len = min(len(clean), len(enhanced))
    clean    = clean[:min_len]
    enhanced = enhanced[:min_len]
    if min_len == 0:
        raise ValueError("Signals must be non-empty")
    noise    = clean - enhanced
    return 10.0 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8))


def calculate_metrics(clean, enhanced, sample_rate=16000):
    stoi_score = calculate_stoi(clean, enhanced, sample_rate)
    pesq_score = calculate_pesq(clean, enhanced, sample_rate)
    snr_db     = calculate_snr(clean, enhanced)
    return stoi_score, pesq_score, snr_db
