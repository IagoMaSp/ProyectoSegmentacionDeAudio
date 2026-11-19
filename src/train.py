"""Comparacion de clasificadores, ablation de caracteristicas y modelo final.

Protocolo:
- Pliegues 0..3: validacion cruzada agrupada (cada pliegue es disjunto en archivos y,
  en musica, en artistas; el audio sintetico de un pliegue sale de ese pliegue).
- Pliegue 4: prueba final, no se toca hasta el final.
- Seleccion por F1 macro promedio en validacion cruzada. Se informa ademas el
  recall de "ruido", la clase minoritaria y la que mas se confunde.
- El modelo elegido se reentrena con los pliegues 0..3 y se evalua una sola vez en
  el pliegue 4. Ese es el modelo que se guarda (los numeros informados son de el).

Uso:  python -m src.train [--rapido] [--modelos svm,rf] [--sin-ablation]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config as C
from src.dataset import load_features_npz
from src.features import FEATURE_GROUPS, get_feature_names, group_indices

LABELS = list(range(len(C.CATEGORIES)))
ABLATION_STEPS = [
    ["base"],
    ["base", "espectrales"],
    ["base", "espectrales", "mfcc"],
    ["base", "espectrales", "mfcc", "deltas"],
    ["base", "espectrales", "mfcc", "deltas", "dinamica"],
    ["mfcc", "deltas"],
]


def make_model(name: str, seed: int = C.RANDOM_STATE) -> Pipeline:
    """Todos con StandardScaler ajustado solo con los datos de entrenamiento del pliegue.

    El dataset se arma balanceado por clase; ademas, los modelos que lo permiten usan
    class_weight="balanced" para que un desbalance residual no castigue a "ruido".
    """
    if name == "knn":
        clf = KNeighborsClassifier(n_neighbors=15, weights="distance")
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                            max_iter=400, random_state=seed)
    elif name == "svm":
        clf = SVC(C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
    elif name == "gb":
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                             l2_regularization=1e-3, class_weight="balanced",
                                             early_stopping=True, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("escalado", StandardScaler()), ("clf", clf)])


MODEL_NAMES = {
    "knn": "k-NN (k=15, ponderado por distancia)",
    "mlp": "MLP (128-64, ReLU)",
    "svm": "SVM RBF (C=10)",
    "rf": "Random Forest (300 arboles)",
    "gb": "Gradient Boosting por histogramas",
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    rec = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "recall": {c: float(r) for c, r in zip(C.CATEGORIES, rec)},
        "f1": {c: float(v) for c, v in zip(C.CATEGORIES, f1c)},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def cross_validate(name: str, X: np.ndarray, y: np.ndarray, folds: np.ndarray, verbose: bool = True) -> dict:
    per_fold, total_conf = [], np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    t0 = time.time()
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        model = make_model(name)
        if name == "svm":
            model.set_params(clf__probability=False)  # en CV solo hace falta la clase
        model.fit(X[tr], y[tr])
        m = metrics(y[va], model.predict(X[va]))
        per_fold.append(m)
        total_conf += np.array(m["confusion"])
    f1s = [m["f1_macro"] for m in per_fold]
    accs = [m["accuracy"] for m in per_fold]
    out = {
        "f1_macro_media": float(np.mean(f1s)),
        "f1_macro_desvio": float(np.std(f1s)),
        "accuracy_media": float(np.mean(accs)),
        "accuracy_desvio": float(np.std(accs)),
        "recall_ruido_media": float(np.mean([m["recall"]["ruido"] for m in per_fold])),
        "recall_media": {c: float(np.mean([m["recall"][c] for m in per_fold])) for c in C.CATEGORIES},
        "confusion_sumada": total_conf.tolist(),
        "segundos": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  {name:4s} F1m {out['f1_macro_media']:.4f} +- {out['f1_macro_desvio']:.4f} | "
              f"acc {out['accuracy_media']:.4f} | recall ruido {out['recall_ruido_media']:.4f} | "
              f"{out['segundos']:.0f} s", flush=True)
    return out


def plot_confusion(conf: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = np.asarray(conf, dtype=np.float64)
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(conf[i, j])})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else "#171717")
    ax.set_xticks(LABELS, C.CATEGORIES)
    ax.set_yticks(LABELS, C.CATEGORIES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_bars(rows: list[tuple[str, float, float]], title: str, path: Path, xlabel: str = "F1 macro (CV)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 0.9))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    ax.barh(names, vals, xerr=errs, color="#171717", height=0.55, error_kw={"ecolor": "#8A8983", "lw": 1})
    lo = max(0.0, min(vals) - 0.08)
    ax.set_xlim(lo, 1.0)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_bundle(model: Pipeline, name: str, X_dev: np.ndarray, y_dev: np.ndarray,
                 feature_idx: np.ndarray, seed: int = C.RANDOM_STATE) -> dict[str, Any]:
    """Todo lo que necesita la inferencia, en un solo archivo."""
    rng = np.random.default_rng(seed)
    scaler: StandardScaler = model.named_steps["escalado"]
    Z = scaler.transform(X_dev)
    pca = PCA(n_components=2, random_state=seed).fit(Z)
    per_class = 500
    sample = np.concatenate([
        rng.choice(np.flatnonzero(y_dev == k), size=min(per_class, int(np.sum(y_dev == k))), replace=False)
        for k in LABELS
    ])
    names = get_feature_names()
    return {
        "modelo": model,
        "nombre": name,
        "descripcion": MODEL_NAMES[name],
        "feature_idx": feature_idx,
        "feature_names": [names[i] for i in feature_idx],
        "categorias": list(C.CATEGORIES),
        "pca": pca,
        "muestra_Z": Z[sample].astype(np.float32),
        "muestra_y": y_dev[sample],
        "muestra_2d": pca.transform(Z[sample]).astype(np.float32),
        "config": {
            "sample_rate": C.SAMPLE_RATE, "n_win": C.N_WIN, "hop": C.HOP_LENGTH, "n_fft": C.N_FFT,
            "hp_fc": C.HP_CUTOFF_HZ, "hp_orden": C.HP_ORDER, "notch": list(C.NOTCH_FREQS_HZ),
            "notch_q": C.NOTCH_Q, "ventana_s": C.SEGMENT_DURATION,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento y comparacion de clasificadores.")
    parser.add_argument("--features", default=str(C.FEATURES_PATH))
    parser.add_argument("--modelos", default="knn,mlp,svm,rf,gb")
    parser.add_argument("--sin-ablation", action="store_true")
    parser.add_argument("--rapido", action="store_true", help="Submuestrea a 1/4 para probar el flujo.")
    args = parser.parse_args()

    d = load_features_npz(args.features)
    X, y, folds, kind = d["X"].astype(np.float64), d["y"], d["fold"], d["kind"]
    if args.rapido:
        sel = np.random.default_rng(0).choice(len(y), len(y) // 4, replace=False)
        X, y, folds, kind = X[sel], y[sel], folds[sel], kind[sel]
    dev = folds != C.TEST_FOLD
    test = ~dev
    print(f"Desarrollo: {dev.sum()} muestras | prueba: {test.sum()} | caracteristicas: {X.shape[1]}")
    all_idx = np.arange(X.shape[1])

    report: dict[str, Any] = {"protocolo": {
        "pliegues_cv": sorted(int(f) for f in np.unique(folds[dev])),
        "pliegue_prueba": C.TEST_FOLD,
        "muestras_desarrollo": int(dev.sum()),
        "muestras_prueba": int(test.sum()),
        "por_clase_desarrollo": dict(zip(C.CATEGORIES, np.bincount(y[dev], minlength=4).tolist())),
        "criterio": "F1 macro promedio en CV agrupada por archivo",
    }}

    print("\nComparacion de modelos (64 caracteristicas, CV de 4 pliegues):")
    cv = {}
    for name in args.modelos.split(","):
        cv[name] = cross_validate(name, X[dev], y[dev], folds[dev])
        plot_confusion(cv[name]["confusion_sumada"], f"{MODEL_NAMES[name]} - CV",
                       C.FIGURES_DIR / f"confusion_cv_{name}.png")
    report["modelos_cv"] = cv
    best = max(cv, key=lambda n: cv[n]["f1_macro_media"])
    print(f"Elegido por F1 macro: {best}")
    plot_bars([(MODEL_NAMES[n], cv[n]["f1_macro_media"], cv[n]["f1_macro_desvio"]) for n in cv],
              "Comparacion de clasificadores (CV agrupada)", C.FIGURES_DIR / "comparacion_modelos.png")

    if not args.sin_ablation:
        print(f"\nAblation de caracteristicas con {best}:")
        abl = []
        for groups in ABLATION_STEPS:
            idx = group_indices(groups)
            r = cross_validate(best, X[dev][:, idx], y[dev], folds[dev], verbose=False)
            label = " + ".join(groups)
            print(f"  {label:45s} ({idx.size:2d}) F1m {r['f1_macro_media']:.4f} | recall ruido {r['recall_ruido_media']:.4f}")
            abl.append({"grupos": groups, "n": int(idx.size), **{k: r[k] for k in (
                "f1_macro_media", "f1_macro_desvio", "accuracy_media", "recall_ruido_media", "recall_media")}})
        report["ablation"] = abl
        plot_bars([(f"{' + '.join(a['grupos'])} ({a['n']})", a["f1_macro_media"], a["f1_macro_desvio"]) for a in abl],
                  f"Ablation de caracteristicas ({MODEL_NAMES[best]})", C.FIGURES_DIR / "ablation.png")

    print("\nEntrenando el modelo final con los pliegues de desarrollo...")
    model = make_model(best)
    model.fit(X[dev], y[dev])
    pred = model.predict(X[test])
    test_m = metrics(y[test], pred)
    def coarse(k: str) -> str:
        if k.startswith("silencio_") and "celular" in k:
            return "silencio_por_celular"
        return k

    kind_test = np.array([coarse(k) for k in kind[test]])
    by_kind = {}
    for k in sorted(set(kind_test)):
        m = kind_test == k
        by_kind[k] = {"n": int(m.sum()), "accuracy": float(np.mean(pred[m] == y[test][m]))}
    report["prueba"] = {"modelo": best, **test_m, "accuracy_por_tipo": by_kind}
    plot_confusion(test_m["confusion"], f"{MODEL_NAMES[best]} - prueba (pliegue {C.TEST_FOLD})",
                   C.FIGURES_DIR / "confusion_prueba.png")
    print(f"Prueba: acc {test_m['accuracy']:.4f} | F1m {test_m['f1_macro']:.4f} | recall {test_m['recall']}")

    bundle = build_bundle(model, best, X[dev], y[dev], all_idx)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, C.MODEL_BUNDLE_PATH, compress=3)
    report["grupos_de_caracteristicas"] = {g: len(v) for g, v in FEATURE_GROUPS.items()}
    report["modelo_guardado"] = str(C.MODEL_BUNDLE_PATH.relative_to(C.BASE_DIR))
    C.METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Modelo en {C.MODEL_BUNDLE_PATH} ({C.MODEL_BUNDLE_PATH.stat().st_size / 1e6:.1f} MB); "
          f"metricas en {C.METRICS_PATH}")


if __name__ == "__main__":
    main()
