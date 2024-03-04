import unittest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.model import ModelHandler

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"


class TestModel(unittest.TestCase):
    """
    Unit tests for the ModelHandler class.
    """

    def test_load_model(self):
        """
        Test whether the load_model method loads the model correctly.
        """
        model = ModelHandler().load_model(str(MODEL_PATH))
        self.assertIsInstance(model, RandomForestRegressor)

    def test_predict(self):
        """
        Test whether the predict method makes predictions using the loaded model.
        """
        model = ModelHandler().load_model(str(MODEL_PATH))
        test_input = pd.DataFrame([[0] * 13], columns=[f'feature_{i}' for i in range(13)])
        prediction = ModelHandler().predict(test_input, model)
        self.assertTrue(len(prediction) > 0)


if __name__ == '__main__':
    unittest.main()
