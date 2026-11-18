"""Construccion del dataset a partir de MUSAN.

1. Cada archivo .wav se asigna a uno de N_FOLDS pliegues, estratificado por clase y
   balanceado por duracion. En musica, todos los archivos de un mismo artista van
   al mismo pliegue (segun los ANNOTATIONS de MUSAN).
2. Por archivo: ganancia aleatoria, acondicionamiento y ventanas de 3 s con la
   misma funcion que usa la inferencia (`features.signal_to_windows`).
3. Por pliegue: silencio sintetico y augmentation (mezclas a SNR, canal de celular)
   construidos SOLO con recortes de archivos de ese pliegue. Asi ninguna muestra de
   validacion comparte audio con una de entrenamiento.

El pliegue TEST_FOLD queda reservado como prueba final; ademas se guarda un banco
de recortes crudos de ese pliegue para las curvas de robustez.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.augment import (
    SILENCE_KINDS,
    apply_gain_db,
    make_silence,
    mix_at_snr,
    phone_channel,
    rms_dbfs,
    synthetic_reverb,
)
from src.config import (
    ACTIVE_MIN_DBFS,
    CATEGORIES,
    CLIP_BANK_PER_CLASS,
    DATA_DIR,
    FEATURES_PATH,
    FILE_GAIN_DB_RANGE,
    LABEL_TO_ID,
    MIX_SNR_DB_RANGE,
    MUSAN_FOLDER_MAP,
    N_FOLDS,
    NOISE_BG_SNR_DB_RANGE,
    NOISE_TRAIN_HOP,
    PER_CLASS_PER_FOLD,
    RANDOM_STATE,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SILENCE_DBFS_RANGE,
    TEST_FOLD,
    TESTBANK_PATH,
    TESTBANK_PER_CLASS,
    TRAIN_HOP,
)
from src.features import extract_segment_features, get_feature_names, signal_to_windows
from src.preprocessing import load_wav

CLIPS_PER_FILE = 8
PAUSE_SECONDS_PER_FILE = 2.0


# ---------------------------------------------------------------------------
# Archivos y pliegues
# ---------------------------------------------------------------------------


def collect_wav_files(data_dir: str | Path = DATA_DIR) -> tuple[list[Path], list[str]]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {base}")
    folders = dict(MUSAN_FOLDER_MAP)
    folders.update({v: v for v in MUSAN_FOLDER_MAP.values()})
    paths, labels = [], []
    for folder, category in folders.items():
        d = base / folder
        if d.exists():
            for p in sorted(d.rglob("*.wav")):
                paths.append(p)
                labels.append(category)
    if not paths:
        raise ValueError(f"No se encontraron archivos .wav en {base}")
    return paths, labels


def read_source_groups(paths: list[Path]) -> list[str]:
    """Agrupa los archivos de musica por artista usando los ANNOTATIONS de MUSAN.

    Formato de music/*/ANNOTATIONS: archivo generos voces(Y/N) artista [compositor].
    El artista se toma global (Kevin_MacLeod aparece en varios subconjuntos), asi un
    mismo artista nunca queda repartido entre entrenamiento y validacion. Los
    ANNOTATIONS de habla solo traen sexo e idioma, sin locutor: ahi cada archivo es
    su propio grupo, igual que en ruido.
    """
    owner: dict[str, str] = {}
    for ann in {p.parent / "ANNOTATIONS" for p in paths}:
        if not ann.exists() or "music" not in ann.parts:
            continue
        for line in ann.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 4:
                owner[parts[0]] = "artista:" + parts[3].lower()
    return [owner.get(p.stem, f"archivo:{p.stem}") for p in paths]


def assign_folds(paths: list[Path], labels: list[str], groups: list[str],
                 n_folds: int = N_FOLDS, seed: int = RANDOM_STATE) -> np.ndarray:
    """Asignacion greedy de grupos a pliegues, por clase, equilibrando bytes de audio."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(paths), dtype=np.int64)
    sizes = np.array([p.stat().st_size for p in paths], dtype=np.float64)
    for category in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == category]
        by_group: dict[str, list[int]] = {}
        for i in idx:
            by_group.setdefault(groups[i], []).append(i)
        names = list(by_group)
        rng.shuffle(names)
        names.sort(key=lambda g: -sizes[by_group[g]].sum())
        load = np.zeros(n_folds)
        for g in names:
            f = int(np.argmin(load))
            folds[by_group[g]] = f
            load[f] += sizes[by_group[g]].sum()
    return folds


# ---------------------------------------------------------------------------
# Procesamiento por archivo (en paralelo)
# ---------------------------------------------------------------------------


def _low_energy_audio(x: np.ndarray, seconds: float, frame: int = 480) -> np.ndarray:
    n = x.size // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = x[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    sel = frames[rms <= np.percentile(rms, 10)]
    return sel.reshape(-1)[: int(seconds * SAMPLE_RATE)].astype(np.float32)


def process_file(args: tuple[str, str, int]) -> dict:
    path, category, seed = args
    rng = np.random.default_rng(seed)
    _, x = load_wav(path, expected_sr=SAMPLE_RATE)
    x = apply_gain_db(x, rng.uniform(*FILE_GAIN_DB_RANGE))
    hop = NOISE_TRAIN_HOP if category == "ruido" else TRAIN_HOP
    X, starts, _ = signal_to_windows(x, hop_seconds=hop, include_tail=False)
    if x.size < SEGMENT_SAMPLES:
        X = X[:0]
        starts = starts[:0]
    # descartar ventanas que en realidad son silencio (pausas largas, colas de pistas)
    keep = []
    for s in starts:
        a = int(round(s * SAMPLE_RATE))
        keep.append(rms_dbfs(x[a:a + SEGMENT_SAMPLES]) > ACTIVE_MIN_DBFS)
    keep = np.array(keep, dtype=bool)
    clips = np.zeros((0, SEGMENT_SAMPLES), dtype=np.float32)
    if x.size >= SEGMENT_SAMPLES:
        n_clips = min(CLIPS_PER_FILE, x.size // SEGMENT_SAMPLES)
        st = rng.integers(0, x.size - SEGMENT_SAMPLES + 1, size=n_clips)
        cand = np.stack([x[s:s + SEGMENT_SAMPLES] for s in st]).astype(np.float32)
        levels = np.array([rms_dbfs(c) for c in cand])
        clips = cand[levels > ACTIVE_MIN_DBFS]
    pauses = _low_energy_audio(x, PAUSE_SECONDS_PER_FILE) if category == "voz" else np.zeros(0, np.float32)
    return {"X": X[keep].astype(np.float32), "clips": clips, "pauses": pauses}


# ---------------------------------------------------------------------------
# Augmentation por pliegue
# ---------------------------------------------------------------------------


def _features_of_clips(clips: np.ndarray, workers: int) -> np.ndarray:
    if len(clips) == 0:
        return np.zeros((0, len(get_feature_names())), dtype=np.float32)
    with ProcessPoolExecutor(workers) as ex:
        feats = list(ex.map(extract_segment_features, list(clips), chunksize=32))
    return np.vstack(feats).astype(np.float32)


def _pick(bank: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return bank[rng.integers(0, len(bank), size=n)].astype(np.float64)


def split_budget(n: int) -> dict[str, int]:
    """Reparto de las muestras sinteticas de una clase: 1/3 mezclas, 1/6 reverb y
    ganancia, el resto canal de celular."""
    return {"mezcla": n // 3, "reverb": n // 6, "celular": n - n // 3 - n // 6}


def synth_fold(banks: dict[str, np.ndarray], pauses: np.ndarray, n_aug: dict[str, int],
               rng: np.random.Generator) -> list[tuple[str, str, np.ndarray]]:
    """Devuelve una lista de (clase, tipo_de_muestra, audio_crudo) para el pliegue."""
    out: list[tuple[str, str, np.ndarray]] = []
    noise = banks["ruido"]
    for cat in ("musica", "voz"):
        bank = banks[cat]
        if len(bank) == 0:
            continue
        n_per_kind = split_budget(n_aug[cat])
        for c in _pick(bank, n_per_kind["mezcla"], rng):
            y = mix_at_snr(c, _pick(noise, 1, rng)[0], rng.uniform(*MIX_SNR_DB_RANGE))
            out.append((cat, "mezcla_ruido", y))
        for c in _pick(bank, n_per_kind["reverb"], rng):
            out.append((cat, "reverb_ganancia", apply_gain_db(synthetic_reverb(c, rng), rng.uniform(-12, 4))))
        phone, tags = phone_channel(_pick(bank, n_per_kind["celular"], rng), rng)
        out.extend((cat, t, y) for t, y in zip(tags, phone))
    if len(noise):
        n_per_kind = split_budget(n_aug["ruido"])
        others = np.concatenate([b for k, b in banks.items() if k in ("musica", "voz") and len(b)])
        for c in _pick(noise, n_per_kind["mezcla"], rng):
            y = mix_at_snr(c, _pick(others, 1, rng)[0], rng.uniform(*NOISE_BG_SNR_DB_RANGE))
            out.append(("ruido", "ruido_con_fondo", y))
        for c in _pick(noise, n_per_kind["reverb"], rng):
            out.append(("ruido", "reverb_ganancia", apply_gain_db(synthetic_reverb(c, rng), rng.uniform(-12, 4))))
        phone, tags = phone_channel(_pick(noise, n_per_kind["celular"], rng), rng)
        out.extend(("ruido", t, y) for t, y in zip(tags, phone))
    # silencio: todos los tipos por igual, y una parte pasada por el canal de celular
    sil = []
    for k in range(n_aug["silencio"]):
        kind = SILENCE_KINDS[k % len(SILENCE_KINDS)]
        sil.append((kind, make_silence(kind, rng, SILENCE_DBFS_RANGE, noise_bank=noise, speech_pauses=pauses)))
    n_phone = len(sil) // 6
    phone, tags = phone_channel(np.stack([s for _, s in sil[:n_phone]]), rng)
    for (kind, _), t, y in zip(sil[:n_phone], tags, phone):
        # el canal no debe sacar al silencio de su rango de nivel
        y = y * 10 ** ((min(rms_dbfs(y), SILENCE_DBFS_RANGE[1]) - rms_dbfs(y)) / 20)
        out.append(("silencio", f"silencio_{kind}_{t}", y))
    out.extend(("silencio", f"silencio_{kind}", y) for kind, y in sil[n_phone:])
    return out


def build_dataset(data_dir: str | Path = DATA_DIR, workers: int | None = None, verbose: bool = True,
                  per_class: int = PER_CLASS_PER_FOLD, seed: int = RANDOM_STATE) -> dict[str, np.ndarray]:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    paths, labels = collect_wav_files(data_dir)
    groups = read_source_groups(paths)
    folds = assign_folds(paths, labels, groups, seed=seed)
    rng = np.random.default_rng(seed)
    if verbose:
        for f in range(N_FOLDS):
            counts = {c: sum(1 for l, ff in zip(labels, folds) if l == c and ff == f) for c in set(labels)}
            print(f"  pliegue {f}: {counts}")

    X_all, y_all, fold_all, group_all, kind_all = [], [], [], [], []
    group_ids = {g: i for i, g in enumerate(sorted(set(groups)))}
    testbank: dict[str, np.ndarray] = {}

    for f in range(N_FOLDS):
        idx = np.flatnonzero(folds == f)
        jobs = [(str(paths[i]), labels[i], int(seed * 1000 + i)) for i in idx]
        if verbose:
            print(f"Pliegue {f}: {len(jobs)} archivos", flush=True)
        clean = {c: [] for c in ("musica", "voz", "ruido")}
        banks = {c: [] for c in ("musica", "voz", "ruido")}
        pauses = []
        with ProcessPoolExecutor(workers) as ex:
            for n_done, (i, res) in enumerate(zip(idx, ex.map(process_file, jobs, chunksize=4)), start=1):
                cat = labels[i]
                if len(res["X"]):
                    clean[cat].append((res["X"], group_ids[groups[i]]))
                if len(res["clips"]):
                    banks[cat].append(res["clips"])
                if res["pauses"].size:
                    pauses.append(res["pauses"])
                if verbose and n_done % 200 == 0:
                    print(f"  {n_done}/{len(jobs)}", flush=True)

        bank_arr = {}
        for cat, lst in banks.items():
            arr = np.concatenate(lst) if lst else np.zeros((0, SEGMENT_SAMPLES), np.float32)
            if len(arr) > CLIP_BANK_PER_CLASS:
                arr = arr[rng.choice(len(arr), CLIP_BANK_PER_CLASS, replace=False)]
            bank_arr[cat] = arr
        pause_arr = np.concatenate(pauses) if pauses else np.zeros(0, np.float32)

        # muestras limpias: hasta el 60 % del presupuesto por clase
        n_clean_max = int(0.6 * per_class)
        n_clean = {}
        for cat in ("musica", "voz", "ruido"):
            if not clean[cat]:
                n_clean[cat] = 0
                continue
            Xc = np.concatenate([x for x, _ in clean[cat]])
            gc = np.concatenate([np.full(len(x), g) for x, g in clean[cat]])
            if len(Xc) > n_clean_max:
                sel = rng.choice(len(Xc), n_clean_max, replace=False)
                Xc, gc = Xc[sel], gc[sel]
            n_clean[cat] = len(Xc)
            X_all.append(Xc)
            y_all.append(np.full(len(Xc), LABEL_TO_ID[cat]))
            fold_all.append(np.full(len(Xc), f))
            group_all.append(gc)
            kind_all.extend(["limpio"] * len(Xc))

        # completar el presupuesto de cada clase con augmentation
        n_aug = {cat: max(per_class - n_clean[cat], 0) for cat in n_clean}
        n_aug["silencio"] = per_class
        synth = synth_fold(bank_arr, pause_arr, n_aug, rng)
        if verbose:
            print(f"  limpias {n_clean} | sinteticas {len(synth)}", flush=True)
        Xs = _features_of_clips(np.stack([s for _, _, s in synth]), workers)
        X_all.append(Xs)
        y_all.append(np.array([LABEL_TO_ID[c] for c, _, _ in synth]))
        fold_all.append(np.full(len(synth), f))
        group_all.append(np.full(len(synth), -1 - f))  # grupo sintetico propio del pliegue
        kind_all.extend(k for _, k, _ in synth)

        if f == TEST_FOLD:
            for cat in ("musica", "voz", "ruido"):
                b = bank_arr[cat]
                testbank[cat] = b[: TESTBANK_PER_CLASS]
            sil = [make_silence(SILENCE_KINDS[k % len(SILENCE_KINDS)], rng, SILENCE_DBFS_RANGE,
                                noise_bank=bank_arr["ruido"], speech_pauses=pause_arr)
                   for k in range(TESTBANK_PER_CLASS)]
            testbank["silencio"] = np.stack(sil).astype(np.float32)

    data = {
        "X": np.vstack(X_all).astype(np.float32),
        "y": np.concatenate(y_all).astype(np.int64),
        "fold": np.concatenate(fold_all).astype(np.int64),
        "group": np.concatenate(group_all).astype(np.int64),
        "kind": np.array(kind_all),
    }
    TESTBANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TESTBANK_PATH, **testbank)
    return data


def save_features_npz(path: str | Path, data: dict[str, np.ndarray]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **data,
        feature_names=np.array(get_feature_names()),
        categories=np.array(CATEGORIES),
        sample_rate=SAMPLE_RATE,
        n_folds=N_FOLDS,
        test_fold=TEST_FOLD,
    )
    return out


def load_features_npz(path: str | Path = FEATURES_PATH) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as d:
        if "X" not in d.files:
            raise ValueError(f"{path} tiene el formato viejo; regenerar con `python -m src.build_features`.")
        return {k: d[k] for k in d.files}
