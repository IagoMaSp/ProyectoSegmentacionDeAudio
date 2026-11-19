import numpy as np

from src import features as Fe
from src.config import HOP_LENGTH, N_FFT, N_WIN, SAMPLE_RATE, SEGMENT_SAMPLES
from src.dataset import process_file
from src.preprocessing import frame_signal, hann_window, n_frames_for


def test_conteo_de_frames():
    assert n_frames_for(SEGMENT_SAMPLES) == 1 + (SEGMENT_SAMPLES - N_WIN) // HOP_LENGTH == 298
    for n in (N_WIN, N_WIN + 1, 16000, 48000, 123457):
        x = np.zeros(n)
        assert frame_signal(x).shape == (n_frames_for(n), N_WIN)


def test_cola_hann_periodica():
    w = hann_window(N_WIN, periodic=True)
    total = np.zeros(N_WIN * 6)
    for start in range(0, total.size - N_WIN + 1, HOP_LENGTH):
        total[start:start + N_WIN] += w
    interior = total[N_WIN:-N_WIN]
    assert np.allclose(interior, N_WIN / (2 * HOP_LENGTH))


def test_hann_simetrica_no_es_cola():
    w = hann_window(N_WIN, periodic=False)
    total = np.zeros(N_WIN * 6)
    for start in range(0, total.size - N_WIN + 1, HOP_LENGTH):
        total[start:start + N_WIN] += w
    interior = total[N_WIN:-N_WIN]
    assert np.ptp(interior) > 1e-4


def test_banco_mel():
    fb = Fe.mel_filterbank()
    assert fb.shape == (40, N_FFT // 2 + 1)
    assert np.all(fb >= 0)
    assert np.all(fb.sum(axis=1) > 0)
    assert np.all(np.diff(np.argmax(fb, axis=1)) >= 0)


def test_dct_ortonormal():
    D = Fe.dct_matrix(40, 40)
    assert np.allclose(D @ D.T, np.eye(40), atol=1e-12)


def test_cantidad_de_caracteristicas():
    X, starts, frames = Fe.signal_to_windows(np.random.default_rng(0).standard_normal(SAMPLE_RATE * 10) * 0.1, 0.5)
    assert X.shape[1] == len(Fe.get_feature_names()) == 64
    assert frames.shape[1] == len(Fe.FRAME_FEATURE_NAMES)
    assert np.isfinite(X).all()
    # ventanas cada 0.5 s y la ultima termina en el final del audio
    assert np.allclose(np.diff(starts[:-1]), 0.5)
    assert abs(starts[-1] + 3.0 - 10.0) < HOP_LENGTH / SAMPLE_RATE


def test_fft_propia_mismas_caracteristicas():
    x = np.random.default_rng(1).standard_normal(SAMPLE_RATE * 4) * 0.1
    a, _, _ = Fe.signal_to_windows(x, 0.5, backend="numpy")
    b, _, _ = Fe.signal_to_windows(x, 0.5, backend="propia")
    assert np.allclose(a, b, rtol=1e-8, atol=1e-8)


def test_equivalencia_entrenamiento_inferencia(tmp_path):
    """El dataset y el segmentador producen exactamente los mismos vectores."""
    from scipy.io import wavfile
    from src.augment import apply_gain_db
    from src.config import FILE_GAIN_DB_RANGE

    rng = np.random.default_rng(2)
    t = np.arange(SAMPLE_RATE * 12) / SAMPLE_RATE
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    pcm = (x * 32767).astype(np.int16)
    path = tmp_path / "musica.wav"
    wavfile.write(path, SAMPLE_RATE, pcm)

    seed = 123
    res = process_file((str(path), "musica", seed))
    # mismo camino que la inferencia: misma senal, misma ganancia, misma funcion
    gain = np.random.default_rng(seed).uniform(*FILE_GAIN_DB_RANGE)
    y = apply_gain_db(pcm.astype(np.float64) / 32768.0, gain)
    X_inf, starts, _ = Fe.signal_to_windows(y, hop_seconds=3.0, include_tail=False)
    assert res["X"].shape == X_inf.shape
    assert np.allclose(res["X"], X_inf.astype(np.float32))


def test_senal_corta_se_completa():
    X, starts, _ = Fe.signal_to_windows(np.random.default_rng(3).standard_normal(8000) * 0.1, 0.5)
    assert X.shape[0] == 1 and starts[0] == 0
