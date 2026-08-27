from unittest.mock import patch

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from .utils import (
    FeatureValidationError,
    FRONTEND_TO_MODEL_MAP,
    SYMPTOM_FEATURES,
    build_feature_array,
    build_feature_dataframe,
    feature_names_for_model,
    predict,
)


class FakeModel:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(SYMPTOM_FEATURES)

    def __init__(self, probability=0.75):
        self.received = None
        self.probability = probability

    def predict(self, features):
        raise AssertionError("predict() must not be used to determine the thresholded prediction")

    def predict_proba(self, features):
        self.received = features
        return np.array([[1 - self.probability, self.probability]])


class DatasetNamedFakeModel(FakeModel):
    feature_names_in_ = np.array([FRONTEND_TO_MODEL_MAP[key] for key in SYMPTOM_FEATURES])


CLINICAL_MODEL_FEATURES = (
    "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI", "Hb(g/dl)",
    "Cycle(R/I)", "Cycle length(days)", "Marraige Status (Yrs)",
    "No. of aborptions", "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
    "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "Hip(inch)", "Waist(inch)",
    "Waist:Hip Ratio", "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)",
    "Vit D3 (ng/mL)", "PRG(ng/mL)", "RBS(mg/dl)", "Weight gain(Y/N)",
    "hair growth(Y/N)", "Skin darkening (Y/N)", "Fast food (Y/N)",
    "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)", "Endometrium (mm)", "Avg_Follicle_Count",
)


class ClinicalModelWithIndependentOrder:
    feature_names_in_ = np.array(CLINICAL_MODEL_FEATURES)


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
        self.assertEqual(
            tuple(features.columns),
            tuple(FRONTEND_TO_MODEL_MAP[key] for key in SYMPTOM_FEATURES),
        )
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

    def test_frontend_keys_are_renamed_to_exact_dataset_model_columns(self):
        model = DatasetNamedFakeModel()

        dataframe = build_feature_dataframe("symptom", self.payload, model=model)

        self.assertEqual(
            tuple(dataframe.columns),
            tuple(FRONTEND_TO_MODEL_MAP[key] for key in SYMPTOM_FEATURES),
        )
        self.assertEqual(dataframe.iloc[0]["Age (yrs)"], 28)
        self.assertEqual(dataframe.iloc[0]["Weight (Kg)"], 65.5)
        self.assertEqual(dataframe.iloc[0]["Cycle length(days)"], 35)

    @patch("ml_engine.utils.load_model")
    def test_predict_returns_probability_shape_and_risk_tier(self, load_model):
        model = FakeModel()
        load_model.return_value = model

        result = predict("symptom", self.payload)

        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["feature_count"], 10)
        self.assertEqual(result["probabilities"], {"0": 0.25, "1": 0.75})
        self.assertEqual(result["positive_probability"], 0.75)
        self.assertEqual(result["risk_tier"], "High Risk")
        self.assertIsInstance(model.received, pd.DataFrame)
        self.assertEqual(model.received.shape, (1, 10))
        self.assertEqual(tuple(model.received.columns), SYMPTOM_FEATURES)

    @patch("ml_engine.utils.load_model")
    def test_prediction_and_risk_tier_use_the_custom_probability_threshold(self, load_model):
        for probability, expected_prediction, expected_risk_tier in (
            (0.19, 0, "Low Risk"),
            (0.20, 1, "Moderate Risk"),
            (0.49, 1, "Moderate Risk"),
            (0.50, 1, "High Risk"),
        ):
            with self.subTest(probability=probability):
                load_model.return_value = FakeModel(probability=probability)

                result = predict("symptom", self.payload)

                self.assertEqual(result["prediction"], expected_prediction)
                self.assertEqual(result["positive_probability"], probability)
                self.assertEqual(result["risk_tier"], expected_risk_tier)

    def test_clinical_model_accepts_its_own_33_column_order(self):
        names = feature_names_for_model("clinical", ClinicalModelWithIndependentOrder())

        self.assertEqual(names, CLINICAL_MODEL_FEATURES)
        self.assertNotEqual(names[:10], SYMPTOM_FEATURES)

    def test_clinical_mapping_requires_explicit_complete_model_contract(self):
        with self.assertRaisesRegex(Exception, "seven unspecified clinical feature names"):
            build_feature_array("clinical", {})
