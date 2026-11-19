import numpy as np
import pytest

from src import augment as A
from src.config import SAMPLE_RATE, SILENCE_DBFS_RANGE
from src.filters import condition_offline


def test_mezcla_a_snr_exacta():
    rng = np.random.default_rng(0)
    s = 0.1 * rng.standard_normal(SAMPLE_RATE)
    r = 0.3 * rng.standard_normal(SAMPLE_RATE)
    for snr in (-5, 0, 10, 20):
        y = A.mix_at_snr(s, r, snr)
        noise = y - s
        assert abs(10 * np.log10(np.mean(s ** 2) / np.mean(noise ** 2)) - snr) < 1e-6


@pytest.mark.parametrize("kind", [k for k in A.SILENCE_KINDS if k not in ("dither", "muy_bajo")])
def test_silencio_en_rango_de_nivel(kind):
    rng = np.random.default_rng(1)
    bank = 0.1 * rng.standard_normal((3, 48000))
    pauses = 0.001 * rng.standard_normal(100000)
    for _ in range(5):
        x = A.make_silence(kind, rng, SILENCE_DBFS_RANGE, noise_bank=bank, speech_pauses=pauses)
        # el nivel se define sobre la senal acondicionada
        assert SILENCE_DBFS_RANGE[0] - 1 <= A.rms_dbfs(condition_offline(x)) <= SILENCE_DBFS_RANGE[1] + 1


def test_dither_es_de_un_lsb():
    x = A.dithered_digital_zero(48000, np.random.default_rng(2))
    assert np.max(np.abs(x)) <= 1.0 / 32768 + 1e-12
    assert np.allclose(x * 32768, np.round(x * 32768))


def test_ruido_rosa_cae_3db_por_octava():
    x = A.colored_noise(1 << 18, 1.0, np.random.default_rng(3))
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SAMPLE_RATE)
    band = lambda lo: P[(f >= lo) & (f < 2 * lo)].sum() / ((f >= lo) & (f < 2 * lo)).sum()
    slope = 10 * np.log10(band(2000) / band(1000))
    assert -3.6 < slope < -2.4


def test_reverb_conserva_nivel():
    x = 0.1 * np.random.default_rng(4).standard_normal(48000)
    y = A.synthetic_reverb(x, np.random.default_rng(5), rt60=0.4, wet=0.5)
    assert abs(np.std(y) - np.std(x)) / np.std(x) < 1e-6


def test_codec_alineado():
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    x = 0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    y = A.codec_roundtrip(x, "aac_64k")
    assert y.size == x.size
    assert A.estimate_lag(x, y, max_lag=200) == 0
