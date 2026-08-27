"""Model loading and strict feature mapping for PCOS risk inference."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent / "models"

SYMPTOM_FEATURES: tuple[str, ...] = (
    "Age",
    "Weight",
    "Height",
    "BMI",
    "Cycle(R/I)",
    "Cycle length",
    "Marraige Status",
    "Weight gain",
    "hair growth",
    "Skin darkening",
)

KNOWN_CLINICAL_FEATURES: tuple[str, ...] = (
    "Follicle No. (R)",
    "Follicle No. (L)",
    "Avg_Follicle_Count",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
    "AMH(ng/mL)",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "FSH/LH",
    "TSH (mIU/L)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "Fast food (Y/N)",
)

MODEL_FILENAMES = {
    "symptom": "symptom_pcos_model.pkl",
    "clinical": "full_pcos_model.pkl",
}


class InferenceError(ValueError):
    """Base exception for invalid inference requests or model configuration."""


class FeatureValidationError(InferenceError):
    """Raised when a payload cannot be mapped to the model's exact feature contract."""


class ModelArtifactError(InferenceError):
    """Raised when a required serialized model is unavailable or invalid."""


def _model_path(assessment_type: str) -> Path:
    try:
        filename = MODEL_FILENAMES[assessment_type]
    except KeyError as exc:
        raise ModelArtifactError(f"Unsupported assessment type: {assessment_type}") from exc
    return MODEL_DIR / filename


@lru_cache(maxsize=2)
def load_model(assessment_type: str) -> Any:
    """Load and cache one of the two pre-trained model artifacts."""
    path = _model_path(assessment_type)
    if not path.is_file():
        raise ModelArtifactError(
            f"Model artifact is missing: {path.name}. "
            "Place the trained artifact in ml_engine/models/ before inference."
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # joblib can raise several pickle-related exceptions.
        raise ModelArtifactError(f"Could not load model artifact {path.name}: {exc}") from exc


def _configured_full_feature_names() -> tuple[str, ...] | None:
    """Read an explicit 33-column contract when a model has no feature metadata."""
    raw = os.getenv("PCOS_FULL_FEATURE_NAMES", "")
    if not raw:
        return None
    try:
        names = json.loads(raw) if raw.lstrip().startswith("[") else raw.split(",")
    except json.JSONDecodeError as exc:
        raise ModelArtifactError("PCOS_FULL_FEATURE_NAMES must be valid JSON or comma-separated names") from exc
    return tuple(str(name).strip() for name in names)


def _feature_names_from_pipeline(model: Any) -> tuple[str, ...] | None:
    """Find feature-name metadata on a fitted estimator or nested pipeline step."""
    candidates = [model]
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        candidates.extend(named_steps.values())
    steps = getattr(model, "steps", None)
    if steps:
        candidates.extend(step for _, step in steps)

    for estimator in candidates:
        model_names = getattr(estimator, "feature_names_in_", None)
        if model_names is not None:
            return tuple(str(name) for name in model_names)
    return None


def feature_names_for_model(assessment_type: str, model: Any | None = None) -> tuple[str, ...]:
    """Return the exact feature order, preferring metadata embedded in the trained model."""
    model_names = _feature_names_from_pipeline(model) if model is not None else None

    if assessment_type == "symptom":
        names = model_names or SYMPTOM_FEATURES
        if len(names) != 10:
            raise ModelArtifactError(f"Symptom model contract must contain exactly 10 features; got {len(names)}")
        if names != SYMPTOM_FEATURES:
            raise ModelArtifactError("Symptom model feature order does not match ML_DATA_DICTIONARY.md")
        return names

    if assessment_type != "clinical":
        raise FeatureValidationError(f"Unsupported assessment type: {assessment_type}")

    names = model_names or _configured_full_feature_names()
    if names is None:
        raise ModelArtifactError(
            "The clinical model does not expose feature_names_in_, and the seven unspecified "
            "clinical feature names are not configured. Set PCOS_FULL_FEATURE_NAMES to the "
            "exact 33-column training order; do not guess the missing names."
        )
    if len(names) != 33:
        raise ModelArtifactError(f"Clinical model contract must contain exactly 33 features; got {len(names)}")
    if names[:10] != SYMPTOM_FEATURES:
        raise ModelArtifactError("Clinical model feature order does not begin with the exact 10 symptom features")
    return names


def _number(value: Any, field: str, integer: bool = False) -> int | float:
    if isinstance(value, bool) or value is None:
        raise FeatureValidationError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureValidationError(f"{field} must be numeric") from exc
    if not np.isfinite(number):
        raise FeatureValidationError(f"{field} must be finite")
    if integer:
        if not number.is_integer():
            raise FeatureValidationError(f"{field} must be an integer")
        return int(number)
    return number


def _binary(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "1"}:
            return 1
        if normalized in {"no", "false", "0"}:
            return 0
    if value in (0, 1):
        return int(value)
    raise FeatureValidationError(f"{field} must be a boolean or Yes/No value")


def _value_for_feature(name: str, value: Any) -> int | float:
    if name == "Cycle(R/I)":
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "regular":
                return 2
            if normalized == "irregular":
                return 4
        if value in (2, 4):
            return int(value)
        raise FeatureValidationError("Cycle(R/I) must be Regular/Irregular or 2/4")
    if name in {"Weight gain", "hair growth", "Skin darkening", "Fast food (Y/N)"}:
        return _binary(value, name)
    return _number(value, name, integer=name in {"Cycle length", "Marraige Status", "Follicle No. (R)", "Follicle No. (L)"})


def build_feature_dataframe(assessment_type: str, payload: Mapping[str, Any], model: Any | None = None) -> pd.DataFrame:
    """Validate and map a JSON object into a named DataFrame in exact model order."""
    if not isinstance(payload, Mapping):
        raise FeatureValidationError("features must be a JSON object")
    expected_feature_names = feature_names_for_model(assessment_type, model)
    expected = set(expected_feature_names)
    provided = set(payload)
    missing = [name for name in expected_feature_names if name not in provided]
    extra = sorted(provided - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected fields: {', '.join(extra)}")
        raise FeatureValidationError("Invalid feature payload (" + "; ".join(details) + ")")

    ordered_features_dict = {
        name: _value_for_feature(name, payload[name])
        for name in expected_feature_names
    }
    return pd.DataFrame([ordered_features_dict], columns=expected_feature_names)


# Kept as a compatibility alias for callers that used the old helper name.
def build_feature_array(assessment_type: str, payload: Mapping[str, Any], model: Any | None = None) -> pd.DataFrame:
    return build_feature_dataframe(assessment_type, payload, model=model)


def _risk_tier(probability: float) -> str:
    if probability < 1 / 3:
        return "low"
    if probability < 2 / 3:
        return "moderate"
    return "high"


def predict(assessment_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run predict and predict_proba, returning JSON-serializable inference results."""
    model = load_model(assessment_type)
    df = build_feature_dataframe(assessment_type, payload, model=model)
    received_features_dict = payload
    ordered_features_dict = df.to_dict(orient="records")[0]

    try:
        raw_probabilities = model.predict_proba(df)
        probabilities = np.asarray(raw_probabilities, dtype=float)

        print(f"--- ML DEBUG: Raw received features ---", flush=True)
        print(received_features_dict, flush=True)
        print(f"--- ML DEBUG: Ordered DataFrame fed to model ---", flush=True)
        print(ordered_features_dict, flush=True)
        print(df, flush=True)
        print(f"--- ML DEBUG: Raw Model Output ---", flush=True)
        print(probabilities, flush=True)

        positive_probability = float(probabilities[0, -1])
        print(f"--- ML DEBUG: Risk Tier Logic ---", flush=True)
        print(
            "if positive_probability < 1 / 3:\n"
            "    risk_tier = 'low'\n"
            "elif positive_probability < 2 / 3:\n"
            "    risk_tier = 'moderate'\n"
            "else:\n"
            "    risk_tier = 'high'",
            flush=True,
        )
        risk_tier = _risk_tier(positive_probability)
        print(f"positive_probability={positive_probability} -> risk_tier={risk_tier}", flush=True)

        prediction = np.asarray(model.predict(df)).reshape(-1)
    except Exception as exc:
        raise InferenceError(f"Model inference failed: {exc}") from exc
    if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] == 0:
        raise InferenceError("Model returned an invalid probability shape")
    if prediction.size != 1:
        raise InferenceError("Model returned an invalid prediction shape")
    classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
    probability_map = {str(label): float(probability) for label, probability in zip(classes, probabilities[0])}
    return {
        "prediction": prediction[0].item() if hasattr(prediction[0], "item") else prediction[0],
        "probabilities": probability_map,
        "positive_probability": positive_probability,
        "risk_tier": risk_tier,
        "feature_count": int(df.shape[1]),
    }


# Public names used by views and tests.
__all__ = [
    "FeatureValidationError",
    "InferenceError",
    "ModelArtifactError",
    "SYMPTOM_FEATURES",
    "KNOWN_CLINICAL_FEATURES",
    "build_feature_array",
    "feature_names_for_model",
    "load_model",
    "predict",
]
