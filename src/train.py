import argparse
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEST_MODEL_PATH,
    CATEGORIES,
    DATA_DIR,
    FEATURES_PATH,
    KNN_MODEL_PATH,
    KNN_NEIGHBORS,
    MLP_HIDDEN_LAYERS,
    MLP_MAX_ITER,
    MLP_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
)
from src.dataset import build_datasets, load_features_npz


def fit_scaler(X_train: np.ndarray, X_test: np.ndarray) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def train_knn(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = KNN_NEIGHBORS,
) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")
    knn.fit(X_train_scaled, y_train)
    return knn


def train_mlp(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = MLP_HIDDEN_LAYERS,
    max_iter: int = MLP_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> MLPClassifier:
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
    )
    mlp.fit(X_train_scaled, y_train)
    return mlp


def evaluate_model(model: Any, X_test_scaled: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_test, y_pred, labels=list(range(len(CATEGORIES))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CATEGORIES))),
        target_names=CATEGORIES,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": conf_mat,
        "report": report,
        "predictions": y_pred,
    }


def save_artifacts(
    scaler: StandardScaler,
    knn_model: KNeighborsClassifier,
    mlp_model: MLPClassifier,
    best_model: Any,
    models_dir: str | Path = MODELS_DIR,
) -> None:
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(knn_model, KNN_MODEL_PATH)
    joblib.dump(mlp_model, MLP_MODEL_PATH)
    joblib.dump(best_model, BEST_MODEL_PATH)


def train_and_compare(
    data_dir: str | Path = DATA_DIR,
    max_segments_per_class: int | None = None,
    models_dir: str | Path = MODELS_DIR,
    features_path: str | Path | None = FEATURES_PATH,
    from_audio: bool = False,
) -> dict[str, Any]:
    if not from_audio and features_path is not None and Path(features_path).exists():
        print(f"Cargando caracteristicas desde {features_path}")
        X_train, X_test, y_train, y_test = load_features_npz(features_path)
    else:
        print(f"Extrayendo caracteristicas desde el audio en {data_dir}")
        X_train, X_test, y_train, y_test = build_datasets(
            data_dir=data_dir,
            max_segments_per_class=max_segments_per_class,
            verbose=True,
        )
    scaler, X_train_scaled, X_test_scaled = fit_scaler(X_train, X_test)

    knn_model = train_knn(X_train_scaled, y_train)
    mlp_model = train_mlp(X_train_scaled, y_train)

    knn_metrics = evaluate_model(knn_model, X_test_scaled, y_test)
    mlp_metrics = evaluate_model(mlp_model, X_test_scaled, y_test)

    if mlp_metrics["f1_macro"] >= knn_metrics["f1_macro"]:
        best_name = "MLPClassifier"
        best_model = mlp_model
    else:
        best_name = "KNeighborsClassifier"
        best_model = knn_model

    save_artifacts(
        scaler=scaler,
        knn_model=knn_model,
        mlp_model=mlp_model,
        best_model=best_model,
        models_dir=models_dir,
    )

    print(f"Muestras de entrenamiento: {X_train.shape[0]} | Muestras de prueba: {X_test.shape[0]}")
    print("\n--- Resultados KNeighborsClassifier ---")
    print(f"Accuracy: {knn_metrics['accuracy']:.4f} | F1 Macro: {knn_metrics['f1_macro']:.4f}")
    print(knn_metrics["report"])

    print("\n--- Resultados MLPClassifier ---")
    print(f"Accuracy: {mlp_metrics['accuracy']:.4f} | F1 Macro: {mlp_metrics['f1_macro']:.4f}")
    print(mlp_metrics["report"])

    print(f"Mejor modelo seleccionado y persistido: {best_name}")

    return {
        "scaler": scaler,
        "knn_model": knn_model,
        "mlp_model": mlp_model,
        "best_model": best_model,
        "best_model_name": best_name,
        "knn_metrics": knn_metrics,
        "mlp_metrics": mlp_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento y comparacion de clasificadores de audio.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--features", type=str, default=str(FEATURES_PATH),
                        help="Archivo .npz con caracteristicas precalculadas (se usa si existe).")
    parser.add_argument("--from-audio", action="store_true",
                        help="Ignora el .npz y extrae las caracteristicas desde los .wav.")
    args = parser.parse_args()
    train_and_compare(
        data_dir=args.data_dir,
        max_segments_per_class=args.max_segments,
        features_path=args.features,
        from_audio=args.from_audio,
    )


if __name__ == "__main__":
    main()
