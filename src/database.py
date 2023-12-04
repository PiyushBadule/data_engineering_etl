import sqlite3

import pandas as pd

DATABASE_NAME = 'housing_data.db'


def create_database():
    """
    Creates a connection to the SQLite database.

    :return: SQLite connection object.
    """
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        return conn
    except sqlite3.Error as e:
        raise Exception(f"Database connection failed: {e}")


def save_to_database(df, conn, table_name='transformed_data'):
    """
    Saves a DataFrame to the specified table in the given database connection.

    :param df: DataFrame to be saved to the database.
    :param conn: SQLite database connection object.
    :param table_name: Name of the table where the DataFrame will be saved.
    """
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    except Exception as e:
        raise Exception(f"Saving data to database failed: {e}")


def load_from_database(conn, table_name='transformed_data'):
    """
    Loads data from the specified table in the given database connection into a DataFrame.

    :param conn: SQLite database connection object.
    :param table_name: Name of the table to load data from.
    :return: DataFrame containing data from the specified table.
    """
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        raise Exception(f"Loading data from database failed: {e}")


def save_predictions_to_database(predictions, conn, table_name='predictions'):
    """
    Appends prediction results to the specified table in the given database connection.

    :param predictions: DataFrame containing prediction results.
    :param conn: SQLite database connection object.
    :param table_name: Name of the table where predictions will be saved.
    """
    try:
        predictions.to_sql(table_name, conn, if_exists='append', index=False)
    except Exception as e:
        raise Exception(f"Saving predictions to database failed: {e}")
