import sqlite3

import pandas as pd

from constants import DATABASE_NAME


class DatabaseHandler:
    """
    Class to handle interactions with a SQLite database.
    """

    def __init__(self, database_name=DATABASE_NAME):
        """
        Initialize DatabaseHandler object.

        :param database_name: Name of the SQLite database.
        """
        self.database_name = database_name
        self.connection = self.create_database()

    def create_database(self):
        """
        Creates a connection to the SQLite database.

        :return: SQLite connection object.
        """
        try:
            conn = sqlite3.connect(self.database_name)
            return conn
        except sqlite3.Error as e:
            raise Exception(f"Database connection failed: {e}")

    def save_to_database(self, df, table_name='transformed_data'):
        """
        Saves a DataFrame to the specified table in the database.

        :param df: DataFrame to be saved to the database.
        :param table_name: Name of the table where the DataFrame will be saved.
        """
        try:
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
        except Exception as e:
            raise Exception(f"Saving data to database failed: {e}")

    def load_from_database(self, table_name='transformed_data'):
        """
        Loads data from the specified table in the database into a DataFrame.

        :param table_name: Name of the table to load data from.
        :return: DataFrame containing data from the specified table.
        """
        try:
            return pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
        except Exception as e:
            raise Exception(f"Loading data from database failed: {e}")

    def save_predictions_to_database(self, predictions, table_name='predictions'):
        """
        Appends prediction results to the specified table in the database.

        :param predictions: DataFrame containing prediction results.
        :param table_name: Name of the table where predictions will be saved.
        """
        try:
            predictions.to_sql(table_name, self.connection, if_exists='append', index=False)
        except Exception as e:
            raise Exception(f"Saving predictions to database failed: {e}")
