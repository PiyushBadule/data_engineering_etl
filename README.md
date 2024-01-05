
# Data Engineering Project: Housing Price Prediction

This project implements a Data Engineering pipeline for processing housing data, training a machine learning model, and predicting housing prices. The code is organized to reflect standard practices in data engineering, with separate modules for database operations, model handling, ETL (Extract, Transform, Load) process, and an additional script for direct predictions.

## Project Structure

```
DataEngineeringProject/
│
├── data/             # Contains the datasets used in the project.
│ └── housing.csv
│
├── logs/             # Log files for monitoring the application's behavior.
│ └── project.log
│
├── models/           # Trained machine learning model files.
│ └── model.joblib
│
├── src/              # Source code for the project's modules.
│ ├── database.py     # Handles database operations.
│ ├── etl.py          # Manages the Extract, Transform, Load process.
│ ├── model.py        # Handles machine learning model operations.
│ └── predictor.py    # Performs predictions using the trained model.
│
├── tests/            # Test cases for ensuring module reliability.
│ ├── test_database.py
│ ├── test_etl.py
│ ├── test_model.py
│ └── test_predictor.py
│
├── requirements.txt   # Required Python packages for the project.
├── logger_config.py   # Configuration for logging.
└── main.py            # The main script to run the entire pipeline.
```

### Components

- **`data/`**: Contains the dataset used in the project.
- **`models/`**: Stores the trained machine learning models.
- **`src/`**: Source code for the project, including scripts for database operations, model handling, ETL processes, and predictions.
- **`tests/`**: Contains test cases for validating the functionality of different components of the project.
- **`main.py`**: The main script that runs the ETL pipeline, trains the model, and can be used for other core functionalities.

## Setup and Running the Project

1. **Prerequisites**:
   - Ensure Python 3.9.12 is installed on your system.
   - Install required Python libraries by running:
     ```bash
     pip install -r requirements.txt
     ```
   - Place your dataset in the `data/` directory and the trained model in the `models/` directory.

2. **Running the Pipeline**:
   - Navigate to the project's root directory.
   - Run `python main.py` to start the ETL process and model predictions.
   - Run `python main.py predict` to start the prediction process using `predictor.py`.

3. **Running Tests**:
   - Execute `python -m unittest discover -s tests` to run the test cases.
  
4. **Logging**:
   - Check the /logs/project.log file for detailed logs of the application's execution, useful for debugging and monitoring.
