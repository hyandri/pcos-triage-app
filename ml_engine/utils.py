"""Model loading and strict feature mapping for PCOS risk inference."""

from __future__ import annotations

import json
import os
import re
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

# Clean keys used by the frontend mapped to the original dataset/model names.
# The model metadata remains authoritative when it is available at runtime.
FRONTEND_TO_MODEL_MAP: dict[str, str] = {
    "Age": "Age (yrs)",
    "Weight": "Weight (Kg)",
    "Height": "Height(Cm)",
    "BMI": "BMI",
    "Cycle(R/I)": "Cycle(R/I)",
    "Cycle length": "Cycle length(days)",
    "Marraige Status": "Marraige Status (Yrs)",
    "Weight gain": "Weight gain(Y/N)",
    "hair growth": "hair growth(Y/N)",
    "Skin darkening": "Skin darkening (Y/N)",
}

FRONTEND_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "Age": ("Age", "Age (yrs)", "Age(yrs)"),
    "Weight": ("Weight", "Weight (Kg)", "Weight(Kg)"),
    "Height": ("Height", "Height(Cm)", "Height (Cm)"),
    "BMI": ("BMI",),
    "Cycle(R/I)": ("Cycle(R/I)", "Cycle (R/I)"),
    "Cycle length": ("Cycle length", "Cycle length(days)", "Cycle length (days)"),
    "Marraige Status": (
        "Marraige Status",
        "Marraige Status (Yrs)",
        "Marraige Status(Yrs)",
        "Marriage Status (Yrs)",
    ),
    "Weight gain": ("Weight gain", "Weight gain(Y/N)", "Weight gain (Y/N)"),
    "hair growth": ("hair growth", "hair growth(Y/N)", "hair growth (Y/N)"),
    "Skin darkening": (
        "Skin darkening",
        "Skin darkening (Y/N)",
        "Skin darkening(Y/N)",
    ),
}

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
        names = model_names or tuple(FRONTEND_TO_MODEL_MAP[key] for key in SYMPTOM_FEATURES)
        if len(names) != 10:
            raise ModelArtifactError(f"Symptom model contract must contain exactly 10 features; got {len(names)}")
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


def _frontend_key_for_model_name(model_name: str) -> str | None:
    """Resolve one trained-model column name back to its clean frontend key."""
    normalized_model_name = re.sub(r"[^a-z0-9]", "", model_name.lower())
    for frontend_key, aliases in FRONTEND_FEATURE_ALIASES.items():
        candidates = aliases + (FRONTEND_TO_MODEL_MAP[frontend_key],)
        if model_name in candidates:
            return frontend_key
        if normalized_model_name in {
            re.sub(r"[^a-z0-9]", "", candidate.lower())
            for candidate in candidates
        }:
            return frontend_key
    return None


def build_feature_dataframe(assessment_type: str, payload: Mapping[str, Any], model: Any | None = None) -> pd.DataFrame:
    """Validate clean frontend keys and return a DataFrame named for the trained model."""
    if not isinstance(payload, Mapping):
        raise FeatureValidationError("features must be a JSON object")
    expected_feature_names = feature_names_for_model(assessment_type, model)

    if assessment_type == "symptom":
        model_to_frontend = {
            model_name: _frontend_key_for_model_name(model_name)
            for model_name in expected_feature_names
        }
        unresolved = [model_name for model_name, frontend_key in model_to_frontend.items() if frontend_key is None]
        if unresolved:
            raise ModelArtifactError(
                "No frontend mapping exists for model feature names: " + ", ".join(unresolved)
            )
        expected_frontend_keys = tuple(model_to_frontend[name] for name in expected_feature_names)
    else:
        model_to_frontend = {name: name for name in expected_feature_names}
        expected_frontend_keys = expected_feature_names

    expected = set(expected_frontend_keys)
    provided = set(payload)
    missing = [name for name in expected_frontend_keys if name not in provided]
    extra = sorted(provided - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected fields: {', '.join(extra)}")
        raise FeatureValidationError("Invalid feature payload (" + "; ".join(details) + ")")

    ordered_features_dict = {
        model_name: _value_for_feature(frontend_key, payload[frontend_key])
        for model_name, frontend_key in model_to_frontend.items()
    }
    return pd.DataFrame([ordered_features_dict], columns=expected_feature_names)


# Kept as a compatibility alias for callers that used the old helper name.
def build_feature_array(assessment_type: str, payload: Mapping[str, Any], model: Any | None = None) -> pd.DataFrame:
    return build_feature_dataframe(assessment_type, payload, model=model)


def _risk_tier(raw_probability: float) -> str:
    """Apply the sensitivity-first clinical threshold policy."""
    if raw_probability < 0.20:
        return "Low Risk"
    if raw_probability < 0.50:
        return "Moderate Risk"
    return "High Risk"


def predict(assessment_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run predict and predict_proba, returning JSON-serializable inference results."""
    model = load_model(assessment_type)
    expected_feature_names = feature_names_for_model(assessment_type, model=model)
    print("--- ML DEBUG: Exact model expected feature names ---", flush=True)
    print(expected_feature_names, flush=True)
    df = build_feature_dataframe(assessment_type, payload, model=model)
    received_features_dict = payload
    ordered_features_dict = df.to_dict(orient="records")[0]

    try:
        raw_probabilities = model.predict_proba(df)
        probabilities = np.asarray(raw_probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] == 0:
            raise InferenceError("Model returned an invalid probability shape")

        print("--- ML DEBUG: Raw received features ---", flush=True)
        print(received_features_dict, flush=True)
        print("--- ML DEBUG: Exact model expected feature names ---", flush=True)
        print(expected_feature_names, flush=True)
        print("--- ML DEBUG: Ordered DataFrame fed to model ---", flush=True)
        print(ordered_features_dict, flush=True)
        print(df, flush=True)
        print("--- ML DEBUG: Raw Model Output ---", flush=True)
        print(probabilities, flush=True)

        raw_probability = float(probabilities[0, -1])
        is_positive = raw_probability >= 0.20
        prediction = 1 if is_positive else 0
        risk_tier = _risk_tier(raw_probability)

        print("--- ML DEBUG: Risk Tier Logic ---", flush=True)
        print(
            "raw_probability < 0.20 -> risk_tier = 'Low Risk'\n"
            "0.20 <= raw_probability < 0.50 -> risk_tier = 'Moderate Risk'\n"
            "raw_probability >= 0.50 -> risk_tier = 'High Risk'\n"
            "is_positive = raw_probability >= 0.20\n"
            "prediction = 1 if is_positive else 0",
            flush=True,
        )
        print(
            f"raw_probability={raw_probability} -> is_positive={is_positive}, "
            f"prediction={prediction}, risk_tier={risk_tier}",
            flush=True,
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(f"Model inference failed: {exc}") from exc

    classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
    probability_map = {str(label): float(probability) for label, probability in zip(classes, probabilities[0])}
    return {
        "prediction": prediction,
        "probabilities": probability_map,
        "positive_probability": raw_probability,
        "risk_tier": risk_tier,
        "feature_count": int(df.shape[1]),
    }


# Public names used by views and tests.
__all__ = [
    "FeatureValidationError",
    "InferenceError",
    "ModelArtifactError",
    "SYMPTOM_FEATURES",
    "FRONTEND_TO_MODEL_MAP",
    "KNOWN_CLINICAL_FEATURES",
    "build_feature_array",
    "build_feature_dataframe",
    "feature_names_for_model",
    "load_model",
    "predict",
]
