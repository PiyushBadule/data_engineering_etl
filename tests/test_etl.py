import unittest
import pandas as pd
import os
from etl import ETLProcessor  # Adjust the import path as needed
from database import DatabaseManager  # Adjust the import path as needed

class TestETL(unittest.TestCase):
    """
    Test cases for verifying the functionality of the ETLProcessor class.
    """

    def setUp(self):
        """
        Set up a temporary database and mock data for testing.
        """
        # Initialize an in-memory database for testing
        self.db_manager = DatabaseManager(database_name=":memory:")
        
        # Create mock data and save to a CSV
        self.mock_csv = 'mock_data.csv'
        self.create_mock_data()

    def create_mock_data(self):
        """
        Creates mock data to simulate the real data and saves it as a CSV file.
        """
        mock_data = {
            # ... your mock data ...
        }
        mock_df = pd.DataFrame(mock_data)
        mock_df.to_csv(self.mock_csv, index=False)

    def test_prepare_data(self):
        """
        Test the ETL process to ensure data is prepared correctly.
        """
        # Initialize ETLProcessor with the path to the mock CSV
        etl_processor = ETLProcessor(data_path=self.mock_csv)

        # Prepare data
        X_train, X_test, y_train, y_test = etl_processor.prepare_data()
        
        # Test if data is split correctly
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

        # Test if the correct columns are created
        expected_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                            'total_bedrooms', 'population', 'households', 'median_income',
                            'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
                            'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
                            'ocean_proximity_NEAR OCEAN']
        self.assertListEqual(list(X_train.columns), expected_columns)

    def tearDown(self):
        """
        Clean up after tests by removing the mock data and closing the database connection.
        """
        os.remove(self.mock_csv)
        self.db_manager.connection.close()

if __name__ == '__main__':
    unittest.main()
