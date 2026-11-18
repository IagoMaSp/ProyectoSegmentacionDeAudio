"""Evaluacion con grabaciones propias (data/real/).

Etiquetas, en orden de prioridad:
1. data/real/etiquetas.csv con columnas archivo,inicio,fin,clase (segundos).
2. Si un archivo no esta en el CSV, la clase sale del nombre: lo que esta antes del
   primer "_", "-" o digito (voz_01.m4a, musica-guitarra.wav, silencio.ogg).

Se compara por celdas de 0.5 s: la etiqueta real de una celda es la del intervalo
que contiene su centro. Se informa el sistema completo (con Viterbi), el mismo sin
suavizado y la version anterior (bloques de 3 s, sin acondicionamiento).

Uso:  python -m scripts.eval_real [--dir data/real]
Salida: docs/resultados/eval_real.json y docs/figuras/real_lineas.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.config import CATEGORIES, FIGURES_DIR, LABEL_TO_ID, REAL_DIR, REPORTS_DIR, SAMPLE_RATE, WINDOW_HOP
from src.preprocessing import load_audio
from src.segmenter import analyze_signal, load_bundle
from scripts.legacy_pipeline import LegacyModel

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".amr", ".3gp", ".webm", ".mp4", ".caf"}
SYNONYMS = {
    "musica": "musica", "música": "musica", "music": "musica", "cancion": "musica",
    "voz": "voz", "habla": "voz", "speech": "voz", "charla": "voz",
    "ruido": "ruido", "noise": "ruido", "ambiente": "ruido",
    "silencio": "silencio", "silence": "silencio", "pieza": "silencio",
}
COLORS = {"musica": "#2563EB", "voz": "#059669", "ruido": "#D97706", "silencio": "#94A3B8"}


def label_from_name(name: str) -> str | None:
    head = re.split(r"[_\-\s\d.]", name.lower(), maxsplit=1)[0]
    return SYNONYMS.get(head)


def read_csv(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list[tuple[float, float, str]]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = SYNONYMS.get(row["clase"].strip().lower())
            if cls is None:
                raise ValueError(f"Clase desconocida en {path}: {row['clase']}")
            out.setdefault(row["archivo"].strip(), []).append((float(row["inicio"]), float(row["fin"]), cls))
    return out


def truth_for_cells(n_cells: int, hop: float, intervals: list[tuple[float, float, str]]) -> np.ndarray:
    centers = (np.arange(n_cells) + 0.5) * hop
    y = np.full(n_cells, -1)
    for a, b, cls in intervals:
        y[(centers >= a) & (centers < b)] = LABEL_TO_ID[cls]
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(REAL_DIR))
    args = parser.parse_args()
    base = Path(args.dir)
    files = sorted(p for p in base.glob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"No hay audios en {base}. Ver {base / 'LEEME.md'}.")
    labels_csv = read_csv(base / "etiquetas.csv")
    bundle = load_bundle()
    legacy = LegacyModel()
    hop = WINDOW_HOP

    rows, Y, P, P_raw, P_old = [], [], [], [], []
    timelines = []
    for path in files:
        x = load_audio(path)
        dur = x.size / SAMPLE_RATE
        if path.name in labels_csv:
            intervals = labels_csv[path.name]
        else:
            cls = label_from_name(path.stem)
            if cls is None:
                print(f"  (se saltea {path.name}: no esta en etiquetas.csv y el nombre no indica la clase)")
                continue
            intervals = [(0.0, dur + 1.0, cls)]
        res = analyze_signal(x, bundle, hop=hop)
        pred = np.array([LABEL_TO_ID[c] for c in res["cells"]["labels"]])
        raw = np.argmax(np.array(res["cells"]["raw"]), axis=1)
        y = truth_for_cells(pred.size, hop, intervals)

        # version anterior: bloques fijos de 3 s; la cola < 3 s queda sin clasificar
        n_blocks = x.size // (3 * SAMPLE_RATE)
        old = np.full(pred.size, -1)
        if n_blocks:
            blocks = x[: n_blocks * 3 * SAMPLE_RATE].reshape(n_blocks, -1)
            pb = legacy.predict_clips(blocks)
            centers = (np.arange(pred.size) + 0.5) * hop
            idx = (centers // 3).astype(int)
            ok = idx < n_blocks
            old[ok] = pb[idx[ok]]

        m = y >= 0
        Y.append(y[m]); P.append(pred[m]); P_raw.append(raw[m]); P_old.append(old[m])
        acc = float(np.mean(pred[m] == y[m])) if m.any() else None
        rows.append({
            "archivo": path.name, "duracion": round(dur, 2),
            "clases": sorted({c for _, _, c in intervals}),
            "accuracy": acc,
            "accuracy_sin_suavizado": float(np.mean(raw[m] == y[m])) if m.any() else None,
            "accuracy_anterior": float(np.mean(old[m] == y[m])) if m.any() else None,
            "segmentos": [(s["start_time"], s["end_time"], s["label"], s["confidence"]) for s in res["segments"]],
        })
        timelines.append((path.name, dur, intervals, res["segments"]))
        print(f"{path.name:32s} {dur:6.1f} s  acierto {acc:.3f}  ->  " +
              ", ".join(f"{a:.1f}-{b:.1f} {l}" for a, b, l, _ in rows[-1]["segmentos"]))

    Y, P, P_raw, P_old = (np.concatenate(v) for v in (Y, P, P_raw, P_old))
    labels = sorted(set(Y.tolist()))
    out = {
        "celda_s": hop,
        "n_celdas": int(Y.size),
        "accuracy": float(np.mean(P == Y)),
        "f1_macro": float(f1_score(Y, P, labels=labels, average="macro", zero_division=0)),
        "accuracy_sin_suavizado": float(np.mean(P_raw == Y)),
        "accuracy_anterior": float(np.mean(P_old == Y)),
        "f1_macro_anterior": float(f1_score(Y, np.where(P_old < 0, (Y + 1) % 4, P_old), labels=labels,
                                            average="macro", zero_division=0)),
        "confusion": confusion_matrix(Y, P, labels=list(range(len(CATEGORIES)))).tolist(),
        "archivos": rows,
        "nota": "La version anterior no clasifica la cola de menos de 3 s; esas celdas cuentan como error.",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval_real.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    plot_timelines(timelines)
    print(f"\nCeldas de {hop} s: {Y.size} | accuracy {out['accuracy']:.3f} (sin suavizado {out['accuracy_sin_suavizado']:.3f}, "
          f"version anterior {out['accuracy_anterior']:.3f}) | F1 macro {out['f1_macro']:.3f}")


def plot_timelines(timelines) -> None:
    n = len(timelines)
    fig, axes = plt.subplots(n, 1, figsize=(8, 0.62 * n + 0.6), squeeze=False)
    for ax, (name, dur, truth, segs) in zip(axes[:, 0], timelines):
        for a, b, c in truth:
            ax.broken_barh([(a, min(b, dur) - a)], (0.55, 0.4), color=COLORS[c])
        for s in segs:
            ax.broken_barh([(s["start_time"], s["end_time"] - s["start_time"])], (0.05, 0.4), color=COLORS[s["label"]])
        ax.set_xlim(0, dur)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.75], ["pred.", "real"], fontsize=7)
        ax.set_title(name, fontsize=7, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "real_lineas.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
