import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from database import DatabaseManager  # Adjust the import path as needed
from model import Model  # Adjust the import path as needed
from etl import ETLProcessor  # Adjust the import path as needed
from predictor import Predictor  # Adjust the import path as needed
from logger_config import setup_logger  # Ensure this is correctly set up

# Initialize the logger
setup_logger()

def main() -> None:
    """
    Main function to orchestrate the data engineering pipeline including data preparation,
    model training, prediction, and evaluation.
    """
    try:
        # Define file paths
        project_root = Path(__file__).parent
        model_path = project_root / "models/model.joblib"
        train_data_path = project_root / "data/housing.csv"

        # Initialize database manager
        db_manager = DatabaseManager(database_name="housing_data.db")
        
        # Initialize and load model
        model = Model(model_path=str(model_path))
        
        # Process data using ETL
        etl_processor = ETLProcessor(data_path=str(train_data_path))
        X_train, X_test, y_train, y_test = etl_processor.prepare_data()
        
        # Train the model or load if already trained
        if not model.model:  # Assuming the load_model sets model to None if it fails
            model.train(X_train, y_train)
            model.save_model(model.model)
        
        # Make predictions on the test set
        predictor = Predictor(model=model)
        predicted_values = predictor.perform_prediction(db_manager, X_test)
        logging.info(f"Predictions: {predicted_values}")

        # Evaluate model performance
        train_error = mean_absolute_error(y_train, model.predict(X_train))
        test_error = mean_absolute_error(y_test, predicted_values)
        logging.info(f'Train error: {train_error}')
        logging.info(f'Test error: {test_error}')

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info('Closing database connection.')
        # Ensure the connection is closed properly
        if db_manager and db_manager.connection:
            db_manager.connection.close()

if __name__ == '__main__':
    main()
