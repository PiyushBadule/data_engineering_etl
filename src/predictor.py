import logging
import pandas as pd
from constants import EXPECTED_COLUMN
from src.database import DatabaseHandler
from src.model import ModelHandler


class PredictionRunner:
    """
    Class to handle running predictions on given input data.
    """

    def given_input_data(self):
        """
        Creates a set of test input data for prediction.

        :return: A tuple of dictionaries, each representing a test data instance.
        """
        data_1 = {
            'longitude': -122.64,
            'latitude': 38.01,
            'housing_median_age': 36.0,
            'total_rooms': 1336.0,
            'total_bedrooms': 258.0,
            'population': 678.0,
            'households': 249.0,
            'median_income': 5.5789,
            'ocean_proximity': 'NEAR OCEAN',
        }

        data_2 = {
            'longitude': -115.73,
            'latitude': 33.35,
            'housing_median_age': 23.0,
            'total_rooms': 1586.0,
            'total_bedrooms': 448.0,
            'population': 338.0,
            'households': 182.0,
            'median_income': 1.2132,
            'ocean_proximity': 'INLAND',
        }

        data_3 = {
            'longitude': -117.96,
            'latitude': 33.89,
            'housing_median_age': 24.0,
            'total_rooms': 1332.0,
            'total_bedrooms': 252.0,
            'population': 625.0,
            'households': 230.0,
            'median_income': 4.4375,
            'ocean_proximity': '<1H OCEAN',
        }

        return data_1, data_2, data_3

    def prepare_input_data(self, input_data):
        """
        Prepares a single instance of input data for prediction, including one-hot encoding and ensuring all expected
        columns are present.

        :param input_data: A dictionary containing the input data.
        :return: A DataFrame ready for prediction.
        """
        try:
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)

            expected_columns = EXPECTED_COLUMN

            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[expected_columns]
            return input_df
        except Exception as e:
            logging.error(f"Error in preparing input data: {e}")
            raise

    def run_prediction(self, model_name):
        """
        Runs predictions on a set of test data, logs the results, and saves predictions to the database.

        :param model_name: Path to the trained model file.
        """
        try:
            model = ModelHandler().load_model(model_name)
            input_data = self.given_input_data()

            all_predictions = []

            logging.info('Performing predictions...')
            for data in input_data:
                predicts_value = ModelHandler().predict(self.prepare_input_data(data), model)
                logging.info(f'Predicted Value: {predicts_value}')
                predictions_df = pd.DataFrame(predicts_value, columns=['Predicted_Value'])
                DatabaseHandler().save_predictions_to_database(predictions_df)

                all_predictions.append(f'Predicted Value: {predicts_value}')  # Append each prediction to the list

            return all_predictions  # Return the list of all prediction DataFrames

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
