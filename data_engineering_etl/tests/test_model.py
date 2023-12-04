import unittest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_engineering_etl.src.model import load_model, predict

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"


class TestModel(unittest.TestCase):
    def test_load_model(self):
        model = load_model(str(MODEL_PATH))
        self.assertIsInstance(model, RandomForestRegressor)

    def test_predict(self):
        model = load_model(str(MODEL_PATH))
        test_input = pd.DataFrame([[0] * 13], columns=[f'feature_{i}' for i in range(13)])
        prediction = predict(test_input, model)
        self.assertTrue(len(prediction) > 0)
