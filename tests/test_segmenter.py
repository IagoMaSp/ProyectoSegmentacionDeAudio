import numpy as np

from src.config import CATEGORIES
from src.segmenter import enforce_min_duration, runs, smooth_median, viterbi, windows_to_cells


def _probs(labels, conf=0.8):
    p = np.full((len(labels), len(CATEGORIES)), (1 - conf) / (len(CATEGORIES) - 1))
    p[np.arange(len(labels)), labels] = conf
    return p


def test_viterbi_elimina_un_parpadeo():
    labels = [0] * 10 + [1] + [0] * 10
    path = viterbi(_probs(labels, 0.7), penalty=6.0)
    assert np.all(path == 0)


def test_viterbi_respeta_cambio_sostenido():
    labels = [0] * 10 + [1] * 10
    path = viterbi(_probs(labels), penalty=6.0)
    assert list(path) == labels


def test_viterbi_sin_penalidad_es_argmax():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=50)
    assert np.array_equal(viterbi(p, penalty=0.0), np.argmax(p, axis=1))


def test_mediana_suaviza():
    labels = [0] * 6 + [2] + [0] * 6
    q = smooth_median(_probs(labels), kernel=5)
    assert np.all(np.argmax(q, axis=1) == 0)


def test_duracion_minima():
    labels = np.array([0] * 6 + [1] * 2 + [0] * 6)
    out = enforce_min_duration(labels, _probs(labels), min_cells=3)
    assert len(runs(out)) == 1


def test_celdas_cubren_la_cola():
    duration = 10.2
    starts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.2])
    probs = np.tile([1.0, 0, 0, 0], (starts.size, 1))
    cells = windows_to_cells(starts, probs, duration, 0.5)
    assert cells.shape[0] == 21  # ceil(10.2 / 0.5)
    assert np.allclose(cells.sum(axis=1), 1.0)


def test_audio_mas_corto_que_la_ventana():
    cells = windows_to_cells(np.array([0.0]), np.array([[0.1, 0.7, 0.1, 0.1]]), 1.3, 0.5)
    assert cells.shape[0] == 3 and np.all(np.argmax(cells, axis=1) == 1)
