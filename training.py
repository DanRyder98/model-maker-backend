"""Train a small regression network and report progress to the web app."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when a training job cannot be started or finished."""


@dataclass
class TrainingJob:
    model_id: str
    file_bytes: bytes
    features: list[str]
    target: str
    learning_rate: float
    epochs: int
    hidden_layers: int
    neurons_per_layer: int
    nextjs_url: str
    callback_secret: str


def build_job(
    *,
    file_bytes: bytes,
    features_raw: str,
    target: str,
    config_raw: str,
    model_id: str,
    nextjs_url: str,
    callback_secret: str,
) -> TrainingJob:
    if not model_id.strip():
        raise TrainingError("A model id is required.")
    if not file_bytes:
        raise TrainingError("The uploaded file is empty.")

    features = _parse_features(features_raw)
    config = _parse_config(config_raw)
    target_name = target.strip()
    if not target_name:
        raise TrainingError("A target column is required.")
    if target_name in features:
        raise TrainingError("The target column cannot also be an input feature.")

    return TrainingJob(
        model_id=model_id.strip(),
        file_bytes=file_bytes,
        features=features,
        target=target_name,
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        hidden_layers=config["hidden_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        nextjs_url=nextjs_url,
        callback_secret=callback_secret,
    )


def run_training(job: TrainingJob) -> None:
    try:
        metrics = _fit(job)
    except TrainingError as exc:
        logger.warning("Training failed for %s: %s", job.model_id, exc)
        notify(
            job.nextjs_url,
            job.callback_secret,
            {"modelId": job.model_id, "status": "failed", "error": str(exc)},
        )
        return
    except Exception:
        logger.exception("Unexpected training failure for %s", job.model_id)
        notify(
            job.nextjs_url,
            job.callback_secret,
            {
                "modelId": job.model_id,
                "status": "failed",
                "error": "Training failed because of an unexpected error.",
            },
        )
        return

    notify(
        job.nextjs_url,
        job.callback_secret,
        {"modelId": job.model_id, "status": "completed", "metrics": metrics},
    )


def notify(nextjs_url: str, callback_secret: str, payload: dict[str, Any]) -> None:
    url = f"{nextjs_url.rstrip('/')}/api/update-model"
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"X-Callback-Secret": callback_secret},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception(
            "Could not report status '%s' for model %s",
            payload.get("status"),
            payload.get("modelId"),
        )
        return

    logger.info(
        "Reported %s for model %s",
        payload.get("status"),
        payload.get("modelId"),
    )


def _parse_features(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrainingError("Features must be a JSON list of column names.") from exc

    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(name, str) and name.strip() for name in parsed)
    ):
        raise TrainingError("Select at least one input feature.")

    features = [name.strip() for name in parsed]
    if len(features) != len(set(features)):
        raise TrainingError("Feature names must be unique.")
    return features


def _parse_config(raw: str) -> dict[str, float | int]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrainingError("The network configuration must be JSON.") from exc
    if not isinstance(parsed, dict):
        raise TrainingError("The network configuration must be an object.")

    learning_rate = _as_float(parsed.get("learningRate"), "Learning rate")
    epochs = _as_int(parsed.get("epochs"), "Epochs")
    hidden_layers = _as_int(parsed.get("hiddenLayers"), "Hidden layers")
    neurons = _as_int(parsed.get("neuronsPerLayer"), "Neurons per layer")

    if not 0 < learning_rate <= 1:
        raise TrainingError("Learning rate must be greater than 0 and at most 1.")
    if not 1 <= epochs <= 500:
        raise TrainingError("Epochs must be between 1 and 500.")
    if not 1 <= hidden_layers <= 8:
        raise TrainingError("Hidden layers must be between 1 and 8.")
    if not 1 <= neurons <= 256:
        raise TrainingError("Neurons per layer must be between 1 and 256.")

    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "hidden_layers": hidden_layers,
        "neurons_per_layer": neurons,
    }


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{label} must be a number.")
    if isinstance(value, float) and not value.is_integer():
        raise TrainingError(f"{label} must be a whole number.")
    return int(value)


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"{label} must be a number.")
    return float(value)


def _fit(job: TrainingJob) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from tensorflow import keras

    try:
        frame = pd.read_csv(io.BytesIO(job.file_bytes))
    except Exception as exc:
        raise TrainingError("The file could not be read as a CSV.") from exc

    if frame.empty or len(frame) < 10:
        raise TrainingError(
            "Provide at least 10 rows so the data can be split into train, validation, and test sets."
        )

    missing = [name for name in [*job.features, job.target] if name not in frame.columns]
    if missing:
        raise TrainingError(f"These columns are not in the CSV: {', '.join(missing)}.")

    features = frame[job.features].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(frame[job.target], errors="coerce")
    if features.isna().any().any() or target.isna().any():
        raise TrainingError("Feature and target columns must contain only numbers.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        features.to_numpy(dtype=float),
        target.to_numpy(dtype=float),
        test_size=0.4,
        random_state=42,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
    )

    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std[x_std == 0] = 1

    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    model = keras.Sequential([keras.layers.Input(shape=(len(job.features),))])
    for _ in range(job.hidden_layers):
        model.add(
            keras.layers.Dense(job.neurons_per_layer, activation="relu")
        )
    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=job.learning_rate),
        loss="mse",
    )

    class _ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            logs = logs or {}
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            if loss is None or val_loss is None:
                return
            notify(
                job.nextjs_url,
                job.callback_secret,
                {
                    "modelId": job.model_id,
                    "status": "training",
                    "progress": {
                        "current_epoch": epoch + 1,
                        "current_loss": float(loss),
                        "current_val_loss": float(val_loss),
                    },
                },
            )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=job.epochs,
        verbose=0,
        callbacks=[_ProgressCallback()],
    )

    predictions = model.predict(x_test, verbose=0).reshape(-1)
    test_loss = float(model.evaluate(x_test, y_test, verbose=0))
    sample_count = min(5, len(y_test))

    return {
        "training_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "test_loss": test_loss,
        "r2_score": float(r2_score(y_test, predictions)),
        "sample_predictions": [
            {
                "actual": float(y_test[index]),
                "predicted": float(predictions[index]),
            }
            for index in range(sample_count)
        ],
        "training_history": {
            "loss": [float(value) for value in history.history["loss"]],
            "val_loss": [float(value) for value in history.history["val_loss"]],
            "epochs": list(range(1, len(history.history["loss"]) + 1)),
        },
    }
