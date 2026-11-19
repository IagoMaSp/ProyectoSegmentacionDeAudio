"""Extrae las caracteristicas de MUSAN (con augmentation) y las guarda en un .npz.

Uso:  python -m src.build_features [--data-dir data/musan] [--workers 7]

Ademas deja en data/banco_prueba.npz recortes crudos del pliegue de prueba, que usa
scripts/eval_robustness.py.
"""
import argparse
import time

import numpy as np

from src.config import CATEGORIES, DATA_DIR, FEATURES_PATH, PER_CLASS_PER_FOLD
from src.dataset import build_dataset, save_features_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el .npz de caracteristicas a partir de MUSAN.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output", type=str, default=str(FEATURES_PATH))
    parser.add_argument("--per-class", type=int, default=PER_CLASS_PER_FOLD,
                        help="Muestras por clase y por pliegue.")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    data = build_dataset(args.data_dir, workers=args.workers, per_class=args.per_class)
    out = save_features_npz(args.output, data)
    counts = np.bincount(data["y"], minlength=len(CATEGORIES))
    print("Muestras por clase:", dict(zip(CATEGORIES, counts.tolist())))
    print(f"X: {data['X'].shape} | guardado en {out} ({out.stat().st_size / 1e6:.1f} MB) "
          f"en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
