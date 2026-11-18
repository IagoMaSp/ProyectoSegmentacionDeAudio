import numpy as np
import pytest

from src import filters as F
from src.config import HP_CUTOFF_HZ, SAMPLE_RATE

FS = SAMPLE_RATE


def _gain_db(sos, f):
    _, h = F.sos_frequency_response(sos, FS, freqs=np.atleast_1d(np.asarray(f, dtype=float)))
    return 20 * np.log10(np.maximum(np.abs(h), 1e-300))


def test_notch_50hz_menor_a_menos_40db():
    assert _gain_db(F.design_notch(FS, 50.0), 50.0)[0] < -40
    assert _gain_db(F.design_conditioning(FS), 50.0)[0] < -40


def test_notch_tolera_desvio_de_red():
    g = _gain_db(F.design_notch(FS, 50.0), [49.5, 50.5])
    assert np.all(g < -10)


def test_banda_de_paso_plana():
    f = np.geomspace(200, 7800, 400)
    g = _gain_db(F.design_conditioning(FS), f)
    assert np.max(np.abs(g)) < 0.05


def test_pasa_altos_menos_3db_en_fc():
    g = _gain_db(F.design_highpass(FS), HP_CUTOFF_HZ)[0]
    assert abs(g + 3.01) < 0.05


def test_rechazo_de_dc_y_rumble():
    g = _gain_db(F.design_conditioning(FS), [1.0, 20.0])
    assert g[0] < -100 and g[1] < -40


def test_estabilidad():
    assert F.is_stable(F.design_conditioning(FS))
    for fs in (8000, 22050, 44100, 48000):
        assert F.is_stable(F.design_conditioning(fs))


def test_filtfilt_fase_nula():
    t = np.arange(FS * 2) / FS
    x = np.sin(2 * np.pi * 440 * t)
    y = F.condition_offline(x)
    mid = slice(4000, -4000)
    assert np.max(np.abs(y[mid] - x[mid])) < 1e-3


def test_causal_doble_mismo_modulo_que_offline():
    t = np.arange(FS * 4) / FS
    for f0 in (80.0, 120.0, 1000.0):
        x = np.sin(2 * np.pi * f0 * t)
        off = F.condition_offline(x)
        live = F.CausalConditioner()
        y = np.concatenate([live.process(c) for c in np.array_split(x, 50)])
        a = np.std(off[FS:-FS])
        b = np.std(y[2 * FS:])
        assert abs(a - b) / a < 1e-3


def test_causal_independiente_del_troceo():
    x = np.random.default_rng(0).standard_normal(FS)
    a, b = F.CausalConditioner(), F.CausalConditioner()
    y1 = a.process(x)
    y2 = np.concatenate([b.process(c) for c in np.array_split(x, 17)])
    assert np.allclose(y1, y2)


@pytest.mark.parametrize("fs_in", [44100, 48000, 22050, 8000])
def test_resample_conserva_tono_en_banda(fs_in):
    t = np.arange(fs_in) / fs_in
    x = np.sin(2 * np.pi * 1000 * t)
    y = F.resample(x, fs_in, FS)
    assert abs(y.size - FS) <= 1
    assert abs(np.std(y[2000:-2000]) - np.sqrt(0.5)) < 1e-3


def test_antialias_elimina_alias():
    fs_in = 48000
    t = np.arange(fs_in) / fs_in
    x = np.sin(2 * np.pi * 11000 * t)  # por encima de 8 kHz
    naive = F.naive_resample(x, fs_in, FS)
    clean = F.resample(x, fs_in, FS)
    assert np.std(naive) > 0.5  # el tono se pliega a 5 kHz
    assert 20 * np.log10(np.std(clean[2000:-2000]) / np.sqrt(0.5)) < -70


@pytest.mark.parametrize("fs_in", [44100, 48000])
def test_remuestreo_por_bloques_igual_al_offline(fs_in):
    x = np.random.default_rng(1).standard_normal(fs_in * 2)
    ref = F.resample(x, fs_in, FS)
    sr = F.StreamingResampler(fs_in, FS)
    y = np.concatenate([sr.process(c) for c in np.array_split(x, 29)])
    assert y.size > ref.size - 200
    assert np.max(np.abs(ref[: y.size] - y)) < 1e-9
