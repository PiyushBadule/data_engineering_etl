import sqlite3
import pandas as pd
import logging
from logger_config import setup_logger

# Initialize the logger
setup_logger()

class DatabaseManager:
    """
    A class to manage database connections and operations.

    Attributes:
        database_name (str): The name or path of the database.
        connection (sqlite3.Connection): A SQLite database connection.
    """
    
    def __init__(self, database_name: str):
        """
        Initializes the DatabaseManager with the specified database.

        Args:
            database_name (str): The name or path of the database to connect to.
        """
        self.database_name = database_name
        self.connection = self.create_connection()

    def create_connection(self):
        """
        Creates and returns a connection to the database.

        Returns:
            sqlite3.Connection: A connection object to the SQLite database.
        """
        try:
            conn = sqlite3.connect(self.database_name)
            logging.info(f"Database connected: {self.database_name}")
            return conn
        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            return None

    def save_to_database(self, df: pd.DataFrame, table_name: str = 'default_table') -> None:
        """
        Saves a DataFrame to the specified table in the database.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            table_name (str): The name of the table to save the data to. Defaults to 'default_table'.
        """
        try:
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
            logging.info(f"Data saved to {table_name} in database.")
        except Exception as e:
            logging.error(f"Error saving to database: {e}")

    def load_from_database(self, table_name: str = 'default_table') -> pd.DataFrame:
        """
        Loads data from the specified table in the database.

        Args:
            table_name (str): The name of the table to load data from. Defaults to 'default_table'.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data, or None if an error occurs.
        """
        try:
            data = pd.read_sql(f"SELECT * FROM {table_name}", self.connection)
            logging.info(f"Data loaded from {table_name} table.")
            return data
        except Exception as e:
            logging.error(f"Error loading from database: {e}")
            return None
