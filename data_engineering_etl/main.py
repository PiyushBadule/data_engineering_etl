import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

from data_engineering_etl.src.predictor import run_prediction
from src.database import create_database, load_from_database, save_predictions_to_database
from src.etl import prepare_data
from src.model import load_model, predict

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
TRAIN_DATA_PATH = PROJECT_ROOT / "data/housing.csv"


def perform_prediction(conn, model, data):
    """
    Perform predictions on the provided data using the specified model and save the results to the database.

    :param conn: Database connection object.
    :param model: Trained machine learning model.
    :param data: Data on which predictions are to be made.
    :return: DataFrame containing the predicted values.
    """
    predictions = predict(data, model)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Value'])
    save_predictions_to_database(predictions_df, conn)
    return predictions_df


def main():
    """
    Main function to run the data engineering pipeline.
    It includes data preparation, model loading, prediction, and evaluation.
    """
    try:
        conn = create_database()

        logging.info('Preparing the data...')
        X_train, X_test, y_train, y_test = prepare_data(str(TRAIN_DATA_PATH), conn)

        logging.info('Loading the model...')
        model = load_model(str(MODEL_PATH))

        logging.info('Calculating train dataset predictions...')
        y_pred_train = predict(X_train, model)
        y_pred_test = predict(X_test, model)

        logging.info('Evaluating the model...')
        train_error = mean_absolute_error(y_train, y_pred_train)
        test_error = mean_absolute_error(y_test, y_pred_test)
        logging.info(f'Train error: {train_error}')
        logging.info(f'Test error: {test_error}')

        logging.info('Performing predictions...')
        df = load_from_database(conn)
        predicted_values = perform_prediction(conn, model, df)
        logging.info(predicted_values[:5])

        predictions_list = run_prediction(str(MODEL_PATH), conn)
        logging.info(predictions_list)


    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info('Closing database connection.')
        conn.close()


if __name__ == '__main__':
    main()
