import sqlite3
import unittest

import pandas as pd

from src.etl import DataProcessor


class TestETL(unittest.TestCase):
    """
    Unit tests for the DataProcessor class.
    """

    def setUp(self):
        """
        Set up the test environment by creating an in-memory database and creating mock data.
        """
        self.conn = sqlite3.connect(':memory:')  # Using an in-memory database for testing
        self.mock_csv = 'mock_data.csv'
        self.create_mock_data()

    def create_mock_data(self):
        """
        Create mock data by generating a mock dataset including all categories for 'OCEAN_PROXIMITY'.
        """
        mock_data = {
            'LONGITUDE': [-122.23, -122.22, -122.24, -122.25, -122.26],
            'LAT': [37.88, 37.86, 37.85, 37.84, 37.83],
            'MEDIAN_AGE': [41.0, 21.0, 52.0, 30.0, 45.0],
            'ROOMS': [880, 7099, 1467, 2000, 1500],
            'BEDROOMS': [129, 1106, 190, 300, 250],
            'POP': [322, 2401, 496, 850, 700],
            'HOUSEHOLDS': [126, 1138, 177, 200, 180],
            'MEDIAN_INCOME': [8.3252, 8.3014, 7.2574, 6.5000, 5.0000],
            'OCEAN_PROXIMITY': ['NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'],
            'MEDIAN_HOUSE_VALUE': [452600, 358500, 352100, 500000, 550000],
            'AGENCY': ['Agency1', 'Agency2', 'Agency3', 'Agency4', 'Agency5']
        }
        mock_df = pd.DataFrame(mock_data)
        mock_df.to_csv(self.mock_csv, index=False)

    def test_prepare_data(self):
        """
        Test the prepare_data method of the DataProcessor class.
        """
        X_train, X_test, y_train, y_test = DataProcessor().prepare_data(self.mock_csv, self.conn)

        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

        expected_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                            'total_bedrooms', 'population', 'households', 'median_income',
                            'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
                            'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
                            'ocean_proximity_NEAR OCEAN']
        self.assertListEqual(list(X_train.columns), expected_columns)

    def tearDown(self):
        """
        Clean up by removing the mock CSV file and closing the database connection.
        """
        import os
        os.remove(self.mock_csv)
        self.conn.close()


if __name__ == '__main__':
    unittest.main()
