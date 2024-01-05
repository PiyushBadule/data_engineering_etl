import unittest
import pandas as pd
from database import DatabaseManager  # Adjust the import path as needed

class TestDatabase(unittest.TestCase):
    """
    Test cases for verifying the functionality of the DatabaseManager class.
    """

    def setUp(self):
        """
        Set up a temporary database connection for testing.
        """
        # Assuming the DatabaseManager class takes 'database_name' as an argument
        self.db_manager = DatabaseManager(database_name=":memory:")  # Using in-memory database for testing

    def test_create_database(self):
        """
        Test the creation of a database connection.
        """
        self.assertIsInstance(self.db_manager.connection, sqlite3.Connection)

    def test_save_and_load_database(self):
        """
        Test saving data to and loading data from the database.
        """
        # Create sample data
        test_data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(test_data)

        # Save data to the database
        self.db_manager.save_to_database(df, 'test_table')

        # Load data from the database
        loaded_df = self.db_manager.load_from_database('test_table')

        # Compare the original and loaded DataFrames to ensure they are the same
        pd.testing.assert_frame_equal(df, loaded_df)

    def tearDown(self):
        """
        Close the database connection after tests.
        """
        self.db_manager.connection.close()

if __name__ == '__main__':
    unittest.main()
