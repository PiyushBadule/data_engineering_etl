import pandas as pd
from model import Model  # Make sure this path is correct based on your project structure
import logging
from logger_config import setup_logger
from typing import Any, Optional

# Initialize the logger
setup_logger()

class Predictor:
    """
    A class responsible for making predictions using a pre-trained machine learning model.

    Attributes:
        model (Model): An instance of a pre-trained machine learning model.
    """

    def __init__(self, model: Model):
        """
        Initializes the Predictor with a pre-trained model.

        Args:
            model (Model): A pre-trained machine learning model.
        """
        self.model = model

    def prepare_input_data(self, input_data: Any) -> Optional[pd.DataFrame]:
        """
        Prepares the input data for prediction. This method should be implemented to match the preprocessing
        that was done during model training.

        Args:
            input_data (Any): The raw data to be processed into a format suitable for the model.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the processed data, or None if an error occurs.
        """
        try:
            # Implement input data preparation here
            # Example: input_df = pd.DataFrame([input_data])
            logging.info("Input data prepared for prediction.")
            return input_df  # Ensure this variable is defined based on your preprocessing logic
        except Exception as e:
            logging.error(f"Error preparing input data: {e}")
            return None

    def run_prediction(self, input_data: Any) -> Optional[Any]:
        """
        Runs the model's prediction on the prepared input data.

        Args:
            input_data (Any): The raw data to make predictions on.

        Returns:
            Optional[Any]: The predictions made by the model, or None if an error occurs.
        """
        try:
            prepared_data = self.prepare_input_data(input_data)
            if prepared_data is not None:
                predictions = self.model.predict(prepared_data)
                logging.info("Predictions made successfully.")
                return predictions
            else:
                logging.error("Prepared data is None, prediction aborted.")
                return None
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None
