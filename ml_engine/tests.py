from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from .utils import (
    FeatureValidationError,
    SYMPTOM_FEATURES,
    build_feature_array,
    predict,
)


class FakeModel:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(SYMPTOM_FEATURES)

    def __init__(self):
        self.received = None

    def predict(self, features):
        self.received = features
        return np.array([1])

    def predict_proba(self, features):
        self.received = features
        return np.array([[0.25, 0.75]])


class InferenceEngineTests(SimpleTestCase):
    def setUp(self):
        self.payload = {
            "Age": 28,
            "Weight": 65.5,
            "Height": 165,
            "BMI": 24.1,
            "Cycle(R/I)": "Irregular",
            "Cycle length": 35,
            "Marraige Status": 2,
            "Weight gain": "Yes",
            "hair growth": False,
            "Skin darkening": 0,
        }

    def test_build_feature_array_preserves_dictionary_order_contract(self):
        features = build_feature_array("symptom", self.payload)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape, (1, 10))
        self.assertEqual(tuple(features.columns), SYMPTOM_FEATURES)
        np.testing.assert_allclose(
            features,
            [[28, 65.5, 165, 24.1, 4, 35, 2, 1, 0, 0]],
        )

    def test_build_feature_array_rejects_missing_and_unexpected_fields(self):
        invalid_payload = {**self.payload, "Age": None, "unexpected": 1}
        del invalid_payload["BMI"]

        with self.assertRaises(FeatureValidationError) as context:
            build_feature_array("symptom", invalid_payload)

        self.assertIn("missing fields: BMI", str(context.exception))
        self.assertIn("unexpected fields: unexpected", str(context.exception))

    @patch("ml_engine.utils.load_model")
    def test_predict_returns_probability_shape_and_risk_tier(self, load_model):
        model = FakeModel()
        load_model.return_value = model

        result = predict("symptom", self.payload)

        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["feature_count"], 10)
        self.assertEqual(result["probabilities"], {"0": 0.25, "1": 0.75})
        self.assertEqual(result["positive_probability"], 0.75)
        self.assertEqual(result["risk_tier"], "high")
        self.assertIsInstance(model.received, pd.DataFrame)
        self.assertEqual(model.received.shape, (1, 10))
        self.assertEqual(tuple(model.received.columns), SYMPTOM_FEATURES)

    def test_clinical_mapping_requires_explicit_complete_model_contract(self):
        with self.assertRaisesRegex(Exception, "seven unspecified clinical feature names"):
            build_feature_array("clinical", {})
