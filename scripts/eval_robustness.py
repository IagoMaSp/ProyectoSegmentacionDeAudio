"""Curvas de robustez: accuracy contra SNR, frecuencia de muestreo y codec.

Usa los recortes crudos del pliegue de prueba (data/banco_prueba.npz, generado por
`python -m src.build_features`), que el modelo nunca vio. Cada condicion se evalua
con el sistema nuevo y con la version anterior (scripts/legacy_pipeline.py).

Uso:  python -m scripts.eval_robustness [--n 300]
Salida: docs/resultados/robustez.json y docs/figuras/robustez_*.png
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.augment import SILENCE_KINDS, codec_batch, make_silence, mix_at_snr, phone_channel, rms_dbfs
from src.config import (ACTIVE_MIN_DBFS, CATEGORIES, DATA_DIR, FIGURES_DIR, LABEL_TO_ID, REPORTS_DIR, SAMPLE_RATE,
                        SILENCE_DBFS_RANGE, TEST_FOLD, TESTBANK_PATH)
from src.features import signal_to_windows
from src.filters import naive_resample, resample
from src.preprocessing import load_wav
from src.segmenter import analyze_signal, load_bundle, predict_proba
from scripts.legacy_pipeline import LegacyModel

SNRS = [20, 10, 5, 0, -5]
FS_CONDITIONS = [
    ("16000 (original)", None),
    ("44100 -> 16000", ("ida_vuelta", 44100)),
    ("11025", ("ida_vuelta", 11025)),
    ("8000", ("ida_vuelta", 8000)),
    ("6000", ("ida_vuelta", 6000)),
    ("8000 sin anti-alias", ("naive", 8000)),
]
CODEC_CONDITIONS = ["sin codec", "mp3_128k", "mp3_32k", "aac_32k", "opus_16k", "amr_nb", "celular completo"]


SMOOTHING_VARIANTS = [("ninguno", None), ("mediana", None), ("viterbi", 2.0), ("viterbi", 4.0),
                      ("viterbi", 6.0), ("viterbi", 10.0)]


def test_fold_files() -> dict[str, list]:
    """Archivos del pliegue de prueba, con la misma asignacion que build_features."""
    from src.dataset import assign_folds, collect_wav_files, read_source_groups
    paths, labels = collect_wav_files(DATA_DIR)
    folds = assign_folds(paths, labels, read_source_groups(paths))
    return {c: [p for p, l, f in zip(paths, labels, folds) if l == c and f == TEST_FOLD]
            for c in ("musica", "voz", "ruido")}


def make_sequences(clips: dict, n_seq: int, rng: np.random.Generator):
    """Secuencias de 6 a 10 tramos de 3 a 15 s con clase distinta a la anterior.

    Cada tramo es audio CONTINUO de un archivo del pliegue de prueba (no recortes de
    3 s pegados: las uniones entre canciones distintas cada 3 s generan saltos de
    flujo espectral que no existen en una grabacion real). El silencio se sintetiza
    continuo con los mismos 8 tipos del entrenamiento.
    """
    files = test_fold_files()
    cache: dict = {}

    def piece(cls: str, n: int) -> np.ndarray:
        if cls == "silencio":
            kind = SILENCE_KINDS[int(rng.integers(len(SILENCE_KINDS)))]
            return make_silence(kind, rng, SILENCE_DBFS_RANGE, noise_bank=clips["ruido"], n=n)
        for _ in range(20):
            path = files[cls][int(rng.integers(len(files[cls])))]
            if path not in cache:
                cache[path] = load_wav(path)[1]
            x = cache[path]
            if x.size < n:
                continue
            a = int(rng.integers(0, x.size - n + 1))
            seg = x[a:a + n]
            if rms_dbfs(seg) > ACTIVE_MIN_DBFS:
                return seg
        return np.resize(clips[cls][int(rng.integers(len(clips[cls])))], n)

    seqs = []
    for _ in range(n_seq):
        parts, truth, t, prev = [], [], 0.0, None
        for _ in range(int(rng.integers(6, 11))):
            cls = str(rng.choice([c for c in CATEGORIES if c != prev]))
            dur = float(rng.uniform(3.0, 15.0))
            parts.append(piece(cls, int(dur * SAMPLE_RATE)))
            truth.append((t, t + dur, cls))
            t += dur
            prev = cls
        seqs.append((np.concatenate(parts), truth))
    return seqs


def eval_segmentation(bundle, clips: dict, n_seq: int = 30) -> dict:
    seqs = make_sequences(clips, n_seq, np.random.default_rng(3))
    rows = {}
    for mode, pen in SMOOTHING_VARIANTS:
        name = mode if pen is None else f"viterbi lambda={pen:g}"
        acc, n_pred, n_true, bnd = [], 0, 0, []
        for x, truth in seqs:
            kw = {"smoothing": mode, "min_duration": 0.0 if mode == "ninguno" else 1.5}
            if pen is not None:
                kw["penalty"] = pen
            res = analyze_signal(x, bundle, **kw)
            centers = np.array(res["cells"]["t"]) + res["hop"] / 2
            y = np.zeros(centers.size, dtype=int)
            for a, b, c in truth:
                y[(centers >= a) & (centers < b)] = LABEL_TO_ID[c]
            pred = np.array([LABEL_TO_ID[c] for c in res["cells"]["labels"]])
            acc.append(np.mean(pred == y))
            n_pred += len(res["segments"])
            n_true += len(truth)
            pb = np.array([s["start_time"] for s in res["segments"][1:]])
            for a, _, _ in truth[1:]:
                bnd.append(float(np.min(np.abs(pb - a))) if pb.size else 10.0)
        rows[name] = {"accuracy_celdas": float(np.mean(acc)), "segmentos_predichos": n_pred,
                      "segmentos_reales": n_true, "error_borde_mediana_s": float(np.median(bnd)),
                      "error_borde_medio_s": float(np.mean(bnd))}
        print(f"segm. {name:18s}: acc {rows[name]['accuracy_celdas']:.3f} | segmentos {n_pred}/{n_true} | "
              f"borde mediano {rows[name]['error_borde_mediana_s']:.2f} s", flush=True)
    return {"n_secuencias": n_seq, "variantes": rows}


def predict_new(bundle, clips: np.ndarray) -> np.ndarray:
    X = np.vstack([signal_to_windows(c, hop_seconds=3.0, include_tail=False)[0][:1] for c in clips])
    return np.argmax(predict_proba(bundle, X), axis=1)


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def per_class(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {c: float(np.mean(pred[y == k] == k)) for k, c in enumerate(CATEGORIES) if np.any(y == k)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Recortes por clase.")
    args = parser.parse_args()
    if not TESTBANK_PATH.exists():
        raise SystemExit(f"Falta {TESTBANK_PATH}; correr `python -m src.build_features`.")

    rng = np.random.default_rng(7)
    bank = np.load(TESTBANK_PATH)
    clips = {c: bank[c][: args.n].astype(np.float64) for c in CATEGORIES}
    all_clips = np.concatenate([clips[c] for c in CATEGORIES])
    all_y = np.concatenate([np.full(len(clips[c]), LABEL_TO_ID[c]) for c in CATEGORIES])
    bundle = load_bundle()
    legacy = LegacyModel()
    t0 = time.time()

    def run(x: np.ndarray, y: np.ndarray) -> dict:
        pn, pl = predict_new(bundle, x), legacy.predict_clips(x)
        return {"nuevo": accuracy(pn, y), "anterior": accuracy(pl, y),
                "nuevo_por_clase": per_class(pn, y), "anterior_por_clase": per_class(pl, y)}

    out: dict = {"n_por_condicion": int(len(all_y)), "modelo": bundle["descripcion"]}

    # --- SNR: voz y musica con ruido del pliegue de prueba -----------------
    fg = np.concatenate([clips["voz"], clips["musica"]])
    fg_y = np.concatenate([np.full(len(clips["voz"]), LABEL_TO_ID["voz"]),
                           np.full(len(clips["musica"]), LABEL_TO_ID["musica"])])
    noise = clips["ruido"]
    pair = rng.integers(0, len(noise), size=len(fg))
    snr_rows = {"limpio": run(fg, fg_y)}
    for snr in SNRS:
        mixed = np.stack([mix_at_snr(fg[i], noise[pair[i]], snr) for i in range(len(fg))])
        snr_rows[f"{snr} dB"] = run(mixed, fg_y)
        print(f"SNR {snr:>3} dB: nuevo {snr_rows[f'{snr} dB']['nuevo']:.3f} | anterior {snr_rows[f'{snr} dB']['anterior']:.3f}", flush=True)
    out["snr"] = {
        "x": SNRS, "etiquetas": list(snr_rows),
        "nuevo": [r["nuevo"] for r in snr_rows.values()], "anterior": [r["anterior"] for r in snr_rows.values()],
        "detalle": snr_rows,
    }
    # el grafico usa solo los SNR finitos
    out["snr"]["nuevo_curva"] = out["snr"]["nuevo"][1:]
    out["snr"]["anterior_curva"] = out["snr"]["anterior"][1:]

    # --- frecuencia de muestreo ------------------------------------------
    fs_rows = {}
    for name, cond in FS_CONDITIONS:
        if cond is None:
            x = all_clips
        elif cond[0] == "ida_vuelta":
            x = np.stack([resample(resample(c, SAMPLE_RATE, cond[1]), cond[1], SAMPLE_RATE)[: c.size] for c in all_clips])
        else:
            x = np.stack([resample(naive_resample(c, SAMPLE_RATE, cond[1]), cond[1], SAMPLE_RATE)[: c.size] for c in all_clips])
        fs_rows[name] = run(x, all_y)
        print(f"fs {name:22s}: nuevo {fs_rows[name]['nuevo']:.3f} | anterior {fs_rows[name]['anterior']:.3f}", flush=True)
    out["fs"] = {"etiquetas": list(fs_rows), "nuevo": [r["nuevo"] for r in fs_rows.values()],
                 "anterior": [r["anterior"] for r in fs_rows.values()], "detalle": fs_rows}

    # --- codecs ------------------------------------------------------------
    codec_rows = {}
    for name in CODEC_CONDITIONS:
        if name == "sin codec":
            x = all_clips
        elif name == "celular completo":
            x, _ = phone_channel(all_clips, np.random.default_rng(11))
        else:
            x = np.concatenate([codec_batch(clips[c], name) for c in CATEGORIES])
        codec_rows[name] = run(x, all_y)
        print(f"codec {name:16s}: nuevo {codec_rows[name]['nuevo']:.3f} | anterior {codec_rows[name]['anterior']:.3f}", flush=True)
    out["codec"] = {"etiquetas": list(codec_rows), "nuevo": [r["nuevo"] for r in codec_rows.values()],
                    "anterior": [r["anterior"] for r in codec_rows.values()], "detalle": codec_rows}
    out["segmentacion"] = eval_segmentation(bundle, clips)
    out["segundos"] = round(time.time() - t0, 1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "robustez.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    plot(out)
    print(f"Listo en {out['segundos']} s -> {REPORTS_DIR / 'robustez.json'}")


def plot(out: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(SNRS, out["snr"]["nuevo_curva"], "-o", color="#171717", ms=4, label="Sistema nuevo")
    ax.plot(SNRS, out["snr"]["anterior_curva"], "--s", color="#8A8983", ms=4, label="Version anterior")
    ax.axhline(out["snr"]["nuevo"][0], color="#171717", lw=0.6, ls=":")
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.set_xlabel("SNR [dB] (voz o musica + ruido)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustez frente al ruido aditivo (pliegue de prueba)", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(color="#E2E1DA", lw=0.6)
    fig.savefig(FIGURES_DIR / "robustez_snr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for key, title in (("fs", "frecuencia de muestreo"), ("codec", "codec")):
        blk = out[key]
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(blk["etiquetas"]) + 1.0))
        y = np.arange(len(blk["etiquetas"]))
        ax.barh(y - 0.18, blk["nuevo"], height=0.34, color="#171717", label="Sistema nuevo")
        ax.barh(y + 0.18, blk["anterior"], height=0.34, color="#B5B4AC", label="Version anterior")
        ax.set_yticks(y, blk["etiquetas"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy (4 clases)")
        ax.set_title(f"Robustez frente a {title}", fontsize=9)
        ax.legend(frameon=False, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.savefig(FIGURES_DIR / f"robustez_{key}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
