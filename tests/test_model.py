import unittest
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from model import Model  # Adjust the import path as needed

# Define the path to the model
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"

class TestModel(unittest.TestCase):
    """
    Test cases for verifying the functionality of the Model class.
    """

    def setUp(self):
        """
        Set up a Model instance for testing.
        """
        # Initialize the Model with the path to the model file
        self.model = Model(model_path=str(MODEL_PATH))

    def test_load_model(self):
        """
        Test loading the machine learning model.
        """
        # The model is already loaded in setUp
        self.assertIsInstance(self.model.model, RandomForestRegressor)

    def test_predict(self):
        """
        Test making predictions with the model.
        """
        # Create a test input DataFrame
        test_input = pd.DataFrame([[0] * 13], columns=[f'feature_{i}' for i in range(13)])
        
        # Make a prediction using the model
        prediction = self.model.predict(test_input)
        
        # Check if the prediction is successful (i.e., prediction length > 0)
        self.assertTrue(len(prediction) > 0)

if __name__ == '__main__':
    unittest.main()
