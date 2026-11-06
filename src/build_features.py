"""Extrae las caracteristicas del dataset MUSAN y las guarda en un archivo .npz.

Permite entrenar (`python -m src.train`) sin necesitar el dataset de audio original.
"""
import argparse
import time

from src.config import DATA_DIR, FEATURES_PATH
from src.dataset import build_datasets, save_features_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el archivo .npz de caracteristicas a partir de MUSAN.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output", type=str, default=str(FEATURES_PATH))
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    X_train, X_test, y_train, y_test = build_datasets(
        data_dir=args.data_dir,
        max_segments_per_class=args.max_segments,
        verbose=True,
    )
    out = save_features_npz(args.output, X_train, X_test, y_train, y_test, args.max_segments)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Guardado en {out} ({out.stat().st_size / 1e6:.2f} MB) en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
