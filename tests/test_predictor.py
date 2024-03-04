import unittest

from src.predictor import PredictionRunner


class TestPredictor(unittest.TestCase):
    """
    Unit tests for the PredictionRunner class.
    """

    def test_prepare_input_data(self):
        """
        Test the prepare_input_data method of the PredictionRunner class.
        """
        input_data = {'longitude': -122.64, 'latitude': 38.01, 'housing_median_age': 36.0, 'total_rooms': 1336.0,
                      'total_bedrooms': 258.0, 'population': 678.0, 'households': 249.0, 'median_income': 5.5789,
                      'ocean_proximity': 'NEAR OCEAN'}
        prepared_data = PredictionRunner().prepare_input_data(input_data)
        self.assertIsNotNone(prepared_data)
        self.assertIn('ocean_proximity_NEAR OCEAN', prepared_data.columns)


if __name__ == '__main__':
    unittest.main()
