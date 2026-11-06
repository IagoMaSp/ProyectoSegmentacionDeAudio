import numpy as np

from src.config import (
    EPSILON,
    HOP_LENGTH,
    N_WIN,
    SAMPLE_RATE,
    SUBBAND_EDGES,
)
from src.preprocessing import preprocess_segment


def compute_rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(frames), axis=1))


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    sign_changes = np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1])
    return np.mean(sign_changes, axis=1)


def compute_rfft_spectrum(
    windowed_frames: np.ndarray,
    n_win: int = N_WIN,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(n_win, d=1.0 / sr)
    rfft_complex = np.fft.rfft(windowed_frames, n=n_win, axis=1)
    mag_spectrum = np.abs(rfft_complex)
    power_spectrum = np.square(mag_spectrum)
    return freqs, mag_spectrum, power_spectrum


def compute_spectral_centroid(mag_spectrum: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    mag_sum = np.sum(mag_spectrum, axis=1) + EPSILON
    weighted_freqs = np.sum(mag_spectrum * freqs[None, :], axis=1)
    return weighted_freqs / mag_sum


def compute_spectral_bandwidth(
    mag_spectrum: np.ndarray,
    freqs: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    mag_sum = np.sum(mag_spectrum, axis=1) + EPSILON
    freq_diff_sq = np.square(freqs[None, :] - centroids[:, None])
    variance = np.sum(mag_spectrum * freq_diff_sq, axis=1) / mag_sum
    return np.sqrt(variance)


def compute_subband_energy_ratios(
    power_spectrum: np.ndarray,
    freqs: np.ndarray,
    subbands: list[tuple[float, float]] = SUBBAND_EDGES,
) -> np.ndarray:
    total_energy = np.sum(power_spectrum, axis=1) + EPSILON
    n_frames = power_spectrum.shape[0]
    n_bands = len(subbands)
    ratios = np.zeros((n_frames, n_bands), dtype=np.float64)

    for idx, (f_low, f_high) in enumerate(subbands):
        if idx == n_bands - 1:
            mask = (freqs >= f_low) & (freqs <= f_high)
        else:
            mask = (freqs >= f_low) & (freqs < f_high)
        band_energy = np.sum(power_spectrum[:, mask], axis=1)
        ratios[:, idx] = band_energy / total_energy

    return ratios


def extract_frame_features(
    segment: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_win: int = N_WIN,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    _, raw_frames, windowed_frames = preprocess_segment(
        segment,
        n_win=n_win,
        hop_length=hop_length,
    )
    rms = compute_rms(raw_frames)
    zcr = compute_zcr(raw_frames)

    freqs, mag_spectrum, power_spectrum = compute_rfft_spectrum(
        windowed_frames,
        n_win=n_win,
        sr=sr,
    )
    centroids = compute_spectral_centroid(mag_spectrum, freqs)
    bandwidths = compute_spectral_bandwidth(mag_spectrum, freqs, centroids)
    subband_ratios = compute_subband_energy_ratios(power_spectrum, freqs)

    return np.column_stack([rms, zcr, centroids, bandwidths, subband_ratios])


def extract_segment_features(
    segment: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_win: int = N_WIN,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    frame_features = extract_frame_features(
        segment,
        sr=sr,
        n_win=n_win,
        hop_length=hop_length,
    )
    means = np.mean(frame_features, axis=0)
    stds = np.std(frame_features, axis=0)
    return np.concatenate([means, stds])


def get_feature_names() -> list[str]:
    base_names = [
        "rms",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
    ] + [f"subband_ratio_{i + 1}" for i in range(len(SUBBAND_EDGES))]
    mean_names = [f"{name}_mean" for name in base_names]
    std_names = [f"{name}_std" for name in base_names]
    return mean_names + std_names
