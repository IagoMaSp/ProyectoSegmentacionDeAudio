import argparse
from pathlib import Path
from typing import Any
import joblib
import numpy as np

from src.config import (
    BEST_MODEL_PATH,
    ID_TO_LABEL,
    SAMPLE_RATE,
    SCALER_PATH,
    SEGMENT_DURATION,
    SEGMENT_SAMPLES,
)
from src.features import extract_segment_features
from src.preprocessing import load_wav, split_into_segments


def load_inference_artifacts(
    scaler_path: str | Path = SCALER_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
) -> tuple[Any, Any]:
    scaler = joblib.load(Path(scaler_path))
    model = joblib.load(Path(model_path))
    return scaler, model


def segment_signal(
    signal: np.ndarray,
    scaler: Any,
    model: Any,
    sr: int = SAMPLE_RATE,
    segment_samples: int = SEGMENT_SAMPLES,
) -> list[dict[str, Any]]:
    segments = split_into_segments(signal, segment_samples=segment_samples, remove_dc=True)
    if segments.shape[0] == 0:
        return []

    feature_matrix = np.vstack([extract_segment_features(seg, sr=sr) for seg in segments])
    scaled_features = scaler.transform(feature_matrix)
    predictions = model.predict(scaled_features)

    segment_duration = segment_samples / float(sr)
    timeline: list[dict[str, Any]] = []
    for idx, pred_id in enumerate(predictions):
        start_time = idx * segment_duration
        end_time = (idx + 1) * segment_duration
        label = ID_TO_LABEL[int(pred_id)]
        timeline.append(
            {
                "segment_index": idx,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "label_id": int(pred_id),
                "label": label,
            }
        )
    return timeline


def merge_timeline_intervals(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not timeline:
        return []

    merged: list[dict[str, Any]] = [
        {
            "start_time": timeline[0]["start_time"],
            "end_time": timeline[0]["end_time"],
            "label": timeline[0]["label"],
            "label_id": timeline[0]["label_id"],
        }
    ]
    for item in timeline[1:]:
        current = merged[-1]
        if item["label_id"] == current["label_id"]:
            current["end_time"] = item["end_time"]
        else:
            merged.append(
                {
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                    "label": item["label"],
                    "label_id": item["label_id"],
                }
            )
    return merged


def segment_audio_file(
    wav_path: str | Path,
    scaler_path: str | Path = SCALER_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
    merge_consecutive: bool = False,
) -> list[dict[str, Any]]:
    sr, signal = load_wav(wav_path, expected_sr=SAMPLE_RATE)
    scaler, model = load_inference_artifacts(scaler_path=scaler_path, model_path=model_path)
    timeline = segment_signal(
        signal=signal,
        scaler=scaler,
        model=model,
        sr=sr,
        segment_samples=SEGMENT_SAMPLES,
    )
    if merge_consecutive:
        return merge_timeline_intervals(timeline)
    return timeline


def format_timeline(timeline: list[dict[str, Any]]) -> str:
    if not timeline:
        return f"El archivo contiene menos de {SEGMENT_DURATION:.1f} segundos de audio."
    lines = ["Inicio (s) | Fin (s)    | Categoria"]
    lines.append("-" * 38)
    for entry in timeline:
        lines.append(f"{entry['start_time']:10.2f} | {entry['end_time']:10.2f} | {entry['label']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentacion automatica de un archivo .wav.")
    parser.add_argument("wav_path", type=str)
    parser.add_argument("--scaler-path", type=str, default=str(SCALER_PATH))
    parser.add_argument("--model-path", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    timeline = segment_audio_file(
        wav_path=args.wav_path,
        scaler_path=args.scaler_path,
        model_path=args.model_path,
        merge_consecutive=args.merge,
    )
    print(format_timeline(timeline))


if __name__ == "__main__":
    main()
