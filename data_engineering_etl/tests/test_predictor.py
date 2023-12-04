import unittest

from data_engineering_etl.src.predictor import prepare_input_data


class TestPredictor(unittest.TestCase):
    def test_prepare_input_data(self):
        input_data = {'longitude': -122.64, 'latitude': 38.01, 'housing_median_age': 36.0, 'total_rooms': 1336.0,
                      'total_bedrooms': 258.0, 'population': 678.0, 'households': 249.0, 'median_income': 5.5789,
                      'ocean_proximity': 'NEAR OCEAN'}
        prepared_data = prepare_input_data(input_data)
        self.assertIsNotNone(prepared_data)
        self.assertIn('ocean_proximity_NEAR OCEAN', prepared_data.columns)
        # Additional checks for correct data preparation
