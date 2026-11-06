from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    LABEL_TO_ID,
    LOW_ENERGY_PERCENTILE,
    MUSAN_FOLDER_MAP,
    N_WIN,
    RANDOM_STATE,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SILENCE_SCALE_MAX,
    SILENCE_SCALE_MIN,
    TEST_SIZE,
)
from src.features import extract_segment_features
from src.preprocessing import load_wav, remove_dc_offset, split_into_segments


def collect_wav_files(data_dir: str | Path = DATA_DIR) -> tuple[list[Path], list[str]]:
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {base_path}")

    file_paths: list[Path] = []
    labels: list[str] = []

    folder_candidates = dict(MUSAN_FOLDER_MAP)
    for spanish_name in MUSAN_FOLDER_MAP.values():
        folder_candidates[spanish_name] = spanish_name

    for folder_name, category in folder_candidates.items():
        category_dir = base_path / folder_name
        if not category_dir.exists():
            continue
        wavs = sorted(category_dir.rglob("*.wav"))
        for wav_path in wavs:
            file_paths.append(wav_path)
            labels.append(category)

    if not file_paths:
        raise ValueError(f"No se encontraron archivos .wav en {base_path}")

    return file_paths, labels


def split_files_train_test(
    file_paths: list[Path],
    labels: list[str],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[list[Path], list[Path], list[str], list[str]]:
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_files, test_files, train_labels, test_labels


def extract_low_energy_speech_cutouts(
    speech_signals: list[np.ndarray],
    target_count: int,
    segment_samples: int = SEGMENT_SAMPLES,
    frame_size: int = N_WIN,
    percentile: float = LOW_ENERGY_PERCENTILE,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    if not speech_signals or target_count <= 0:
        return []

    low_energy_chunks: list[np.ndarray] = []
    for signal in speech_signals:
        n_frames = signal.shape[0] // frame_size
        if n_frames == 0:
            continue
        frames = signal[: n_frames * frame_size].reshape(n_frames, frame_size)
        rms_per_frame = np.sqrt(np.mean(np.square(frames), axis=1))
        threshold = np.percentile(rms_per_frame, percentile)
        selected = frames[rms_per_frame <= threshold]
        if selected.size > 0:
            low_energy_chunks.append(selected.reshape(-1))

    if not low_energy_chunks:
        return []

    pool = np.concatenate(low_energy_chunks)
    if pool.shape[0] < segment_samples:
        repeats = (segment_samples // pool.shape[0]) + 1
        pool = np.tile(pool, repeats)

    cutouts: list[np.ndarray] = []
    max_start = pool.shape[0] - segment_samples
    for _ in range(target_count):
        start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        segment = remove_dc_offset(pool[start_idx : start_idx + segment_samples].copy())
        cutouts.append(segment)

    return cutouts


def generate_scaled_noise_segments(
    noise_segments: list[np.ndarray],
    target_count: int,
    scale_min: float = SILENCE_SCALE_MIN,
    scale_max: float = SILENCE_SCALE_MAX,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    if not noise_segments or target_count <= 0:
        return []

    scaled_segments: list[np.ndarray] = []
    n_available = len(noise_segments)
    for _ in range(target_count):
        idx = int(rng.integers(0, n_available))
        scale = float(rng.uniform(scale_min, scale_max))
        scaled = remove_dc_offset(noise_segments[idx] * scale)
        scaled_segments.append(scaled)

    return scaled_segments


def generate_silence_segments(
    speech_signals: list[np.ndarray],
    noise_segments: list[np.ndarray],
    target_count: int,
    seed: int = RANDOM_STATE,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_speech_silence = target_count // 2
    n_noise_silence = target_count - n_speech_silence

    speech_cutouts = extract_low_energy_speech_cutouts(
        speech_signals=speech_signals,
        target_count=n_speech_silence,
        rng=rng,
    )
    scaled_noise = generate_scaled_noise_segments(
        noise_segments=noise_segments,
        target_count=n_noise_silence,
        rng=rng,
    )
    silence_segments = speech_cutouts + scaled_noise
    rng.shuffle(silence_segments)
    return silence_segments


def build_split_features(
    file_paths: list[Path],
    labels: list[str],
    max_segments_per_class: int | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    segments_by_class: dict[str, list[np.ndarray]] = {
        "musica": [],
        "voz": [],
        "ruido": [],
    }
    speech_raw_signals: list[np.ndarray] = []

    for wav_path, category in zip(file_paths, labels):
        _, signal = load_wav(wav_path, expected_sr=SAMPLE_RATE)
        if category == "voz":
            speech_raw_signals.append(signal)
        file_segments = split_into_segments(signal, segment_samples=SEGMENT_SAMPLES, remove_dc=True)
        for seg in file_segments:
            segments_by_class[category].append(seg)

    for category in ("musica", "voz", "ruido"):
        class_segments = segments_by_class[category]
        if max_segments_per_class is not None and len(class_segments) > max_segments_per_class:
            indices = rng.choice(len(class_segments), size=max_segments_per_class, replace=False)
            segments_by_class[category] = [class_segments[int(i)] for i in indices]

    counts = [len(segments_by_class[c]) for c in ("musica", "voz", "ruido") if len(segments_by_class[c]) > 0]
    target_silence_count = int(np.mean(counts)) if counts else 0
    if max_segments_per_class is not None:
        target_silence_count = min(target_silence_count, max_segments_per_class)

    segments_by_class["silencio"] = generate_silence_segments(
        speech_signals=speech_raw_signals,
        noise_segments=segments_by_class["ruido"],
        target_count=target_silence_count,
        seed=seed,
    )

    feature_rows: list[np.ndarray] = []
    target_ids: list[int] = []

    for category, seg_list in segments_by_class.items():
        label_id = LABEL_TO_ID[category]
        for seg in seg_list:
            feat_vec = extract_segment_features(seg, sr=SAMPLE_RATE)
            feature_rows.append(feat_vec)
            target_ids.append(label_id)

    X = np.vstack(feature_rows) if feature_rows else np.empty((0, 18), dtype=np.float64)
    y = np.array(target_ids, dtype=np.int64)
    return X, y


def build_datasets(
    data_dir: str | Path = DATA_DIR,
    test_size: float = TEST_SIZE,
    max_segments_per_class: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_paths, labels = collect_wav_files(data_dir)
    train_files, test_files, train_labels, test_labels = split_files_train_test(
        file_paths=file_paths,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, y_train = build_split_features(
        file_paths=train_files,
        labels=train_labels,
        max_segments_per_class=max_segments_per_class,
        seed=random_state,
    )
    X_test, y_test = build_split_features(
        file_paths=test_files,
        labels=test_labels,
        max_segments_per_class=max_segments_per_class,
        seed=random_state + 1,
    )
    return X_train, X_test, y_train, y_test
