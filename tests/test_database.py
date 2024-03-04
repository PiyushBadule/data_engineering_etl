import sqlite3
import unittest

import pandas as pd

from src.database import DatabaseHandler


class TestDatabase(unittest.TestCase):
    """
    Unit tests for the DatabaseHandler class.
    """

    def setUp(self):
        """
        Set up the test environment by creating a database connection.
        """
        self.conn = DatabaseHandler().create_database()

    def test_create_database(self):
        """
        Test whether the create_database method returns a valid SQLite connection object.
        """
        self.assertIsInstance(self.conn, sqlite3.Connection)

    def test_save_and_load_database(self):
        """
        Test whether data can be saved to and loaded from the database correctly.
        """
        test_data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(test_data)
        DatabaseHandler().save_to_database(df, self.conn, 'test_table')
        loaded_df = DatabaseHandler().load_from_database(self.conn, 'test_table')
        pd.testing.assert_frame_equal(df, loaded_df)

    def tearDown(self):
        """
        Clean up by closing the database connection.
        """
        self.conn.close()


if __name__ == '__main__':
    unittest.main()
