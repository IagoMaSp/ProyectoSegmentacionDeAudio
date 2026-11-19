import numpy as np
import pytest

from src.fft import dft_direct, fft_radix2, rfft_radix2


@pytest.mark.parametrize("n", [2, 8, 64, 512, 4096])
def test_fft_igual_a_numpy(n):
    x = np.random.default_rng(n).standard_normal((3, n)) + 1j * np.random.default_rng(n + 1).standard_normal((3, n))
    assert np.allclose(fft_radix2(x), np.fft.fft(x, axis=-1), atol=1e-9)


def test_fft_igual_a_dft_directa():
    x = np.random.default_rng(3).standard_normal(128)
    assert np.allclose(fft_radix2(x), dft_direct(x), atol=1e-9)


def test_rfft_con_relleno_de_ceros():
    x = np.random.default_rng(4).standard_normal((5, 480))
    assert np.allclose(rfft_radix2(x, n=512), np.fft.rfft(x, n=512, axis=1), atol=1e-9)


def test_largo_no_potencia_de_dos():
    with pytest.raises(ValueError):
        fft_radix2(np.zeros(100))


def test_parseval():
    x = np.random.default_rng(5).standard_normal(1024)
    X = fft_radix2(x)
    assert np.isclose(np.sum(x ** 2), np.sum(np.abs(X) ** 2) / x.size)
