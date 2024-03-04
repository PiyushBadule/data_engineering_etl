import logging
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error

from constants import TRAIN_DATA_PATH, MODEL_PATH
from src.database import DatabaseHandler
from src.etl import DataProcessor
from src.model import ModelHandler
from src.predictor import PredictionRunner

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class DataPipeline:
    """
    Class to run the data engineering pipeline.
    It includes data preparation, model loading, prediction, and evaluation.
    """

    def __init__(self):
        """
        Initialize DataPipeline object.
        """
        self.model = None
        self.conn = None

    def prepare_data(self):
        """
        Prepare the data for training and testing.

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        self.conn = DatabaseHandler().create_database()
        return DataProcessor().prepare_data(str(TRAIN_DATA_PATH))

    def load_model(self):
        """
        Load the trained machine learning model.

        Returns:
            model: Trained machine learning model
        """
        return ModelHandler().load_model(str(MODEL_PATH))

    def evaluate_model(self, y_train, y_test, y_pred_train, y_pred_test):
        """
        Evaluate the trained model.

        Args:
            y_train (array-like): True values of the target variable for the training set.
            y_test (array-like): True values of the target variable for the test set.
            y_pred_train (array-like): Predicted values for the training set.
            y_pred_test (array-like): Predicted values for the test set.
        """
        train_error = mean_absolute_error(y_train, y_pred_train)
        test_error = mean_absolute_error(y_test, y_pred_test)
        logging.info(f'Train error: {train_error}')
        logging.info(f'Test error: {test_error}')

    def perform_prediction(self, data):
        """
        Perform predictions on the provided data and save the results to the database.

        Args:
            data (DataFrame): Data to perform predictions on.

        Returns:
            DataFrame: Predicted values.
        """
        predictions = ModelHandler().predict(data, self.model)
        predictions_df = pd.DataFrame(predictions, columns=['Predicted_Value'])
        DatabaseHandler().save_predictions_to_database(predictions_df)
        return predictions_df

    def run_pipeline(self):
        """
        Run the data engineering pipeline.
        """
        try:
            X_train, X_test, y_train, y_test = self.prepare_data()

            logging.info('Loading the model...')
            self.model = self.load_model()

            logging.info('Calculating train dataset predictions...')
            y_pred_train = ModelHandler().predict(X_train, self.model)
            y_pred_test = ModelHandler().predict(X_test, self.model)

            logging.info('Evaluating the model...')
            self.evaluate_model(y_train, y_test, y_pred_train, y_pred_test)

            logging.info('Performing predictions...')
            df = DatabaseHandler().load_from_database()
            predicted_values = self.perform_prediction(df)
            logging.info(predicted_values[:5])

            predictions_list = PredictionRunner().run_prediction(str(MODEL_PATH))
            logging.info(predictions_list)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            logging.info('Closing database connection.')
            self.conn.close()


if __name__ == '__main__':
    pipeline = DataPipeline()
    pipeline.run_pipeline()
