import joblib
from sklearn.ensemble import RandomForestRegressor
import logging
from logger_config import setup_logger
from typing import Any, Optional

# Initialize the logger
setup_logger()

class Model:
    """
    A class representing a machine learning model for training and prediction.

    Attributes:
        model_path (str): The file path for saving or loading the model.
        model (RandomForestRegressor or Any): The loaded or trained machine learning model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the Model with the specified path.

        Args:
            model_path (str): The file path for saving or loading the model.
        """
        self.model_path = model_path
        self.model = self.load_model()

    def train(self, X_train: Any, y_train: Any) -> Optional[RandomForestRegressor]:
        """
        Trains the RandomForestRegressor model using the provided training data.

        Args:
            X_train (Any): The feature set for training.
            y_train (Any): The target values for training.

        Returns:
            RandomForestRegressor or None: The trained RandomForestRegressor model or None if training fails.
        """
        try:
            regr = RandomForestRegressor(max_depth=12)
            regr.fit(X_train, y_train)
            logging.info("Model training completed successfully.")
            return regr
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            return None

    def predict(self, X: Any) -> Optional[Any]:
        """
        Makes predictions using the trained model on the provided data.

        Args:
            X (Any): The data to make predictions on.

        Returns:
            Any or None: The predicted values or None if prediction fails.
        """
        try:
            predictions = self.model.predict(X)
            logging.info("Predictions made successfully.")
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

    def save_model(self, model: RandomForestRegressor) -> None:
        """
        Saves the provided model to the specified file path.

        Args:
            model (RandomForestRegressor): The model to be saved.
        """
        try:
            joblib.dump(model, self.model_path, compress=3)
            logging.info(f"Model saved to {self.model_path} successfully.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self) -> Optional[RandomForestRegressor]:
        """
        Loads a RandomForestRegressor model from the specified file path.

        Returns:
            RandomForestRegressor or None: The loaded RandomForestRegressor model or None if loading fails.
        """
        try:
            model = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path} successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
