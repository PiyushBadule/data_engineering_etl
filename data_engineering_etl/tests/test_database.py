import sqlite3
import unittest

import pandas as pd

from data_engineering_etl.src.database import create_database, save_to_database, load_from_database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.conn = create_database()

    def test_create_database(self):
        self.assertIsInstance(self.conn, sqlite3.Connection)

    def test_save_and_load_database(self):
        test_data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(test_data)
        save_to_database(df, self.conn, 'test_table')
        loaded_df = load_from_database(self.conn, 'test_table')
        pd.testing.assert_frame_equal(df, loaded_df)

    def tearDown(self):
        self.conn.close()
