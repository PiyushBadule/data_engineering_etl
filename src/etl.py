import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from logger_config import setup_logger
from typing import Tuple, Optional

# Initialize the logger
setup_logger()

class ETLProcessor:
    """
    A class to handle the Extract, Transform, Load (ETL) process for data preparation.

    Attributes:
        data_path (str): The file path to the dataset.
    """

    def __init__(self, data_path: str):
        """
        Initializes the ETLProcessor with the specified data path.

        Args:
            data_path (str): The file path to the dataset.
        """
        self.data_path = data_path

    def prepare_data(self, test_size: float = 0.2, random_state: Optional[int] = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepares the data for modeling by performing extraction, transformation, and splitting.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int, optional): The seed used by the random number generator.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing split data (X_train, X_test, y_train, y_test).
        """
        try:
            # Read the data
            df = pd.read_csv(self.data_path)

            # Implement data cleaning and processing here
            # Example: df = df.dropna()

            # Assuming 'Target' is the column you want to predict
            X = df.drop(columns=['Target'])
            y = df['Target']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            logging.info("Data preparation completed successfully.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            # Return None in case of an error
            return None, None, None, None
