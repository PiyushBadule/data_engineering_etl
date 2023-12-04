
# Data Engineering Project: Housing Price Prediction

This project implements a Data Engineering pipeline for processing housing data, training a machine learning model, and predicting housing prices. The code is organized to reflect standard practices in data engineering, with separate modules for database operations, model handling, ETL (Extract, Transform, Load) process, and an additional script for direct predictions.

## Project Structure

```
data_engineering_project/
│
├── data/
│   └── housing.csv          # Your data file
│
├── models/
│   └── model.joblib         # Your trained model file
│
├── src/
│   ├── __init__.py
│   ├── database.py          # Functions for database operations
│   ├── model.py             # Functions for model training and prediction
│   ├── etl.py               # ETL (Extract, Transform, Load) pipeline functions
│   └── predictor.py         # Script for making predictions with user input
│
├── tests/
│   ├── test_database.py     # Test cases for database operations
│   ├── test_model.py        # Test cases for model functionalities
│   ├── test_etl.py          # Test cases for ETL processes
│   └── test_predictor.py    # Test cases for the prediction script
│
└── main.py                  # Main script to run the ETL pipeline or make predictions
```

### Components

- **`data/`**: Contains the dataset used in the project.
- **`models/`**: Stores the trained machine learning models.
- **`src/`**: Source code for the project, including scripts for database operations, model handling, ETL processes, and predictions.
- **`tests/`**: Contains test cases for validating the functionality of different components of the project.
- **`main.py`**: The main script that runs the ETL pipeline, trains the model, and can be used for other core functionalities.

## Setup and Running the Project

1. **Prerequisites**:
   - Ensure Python is installed on your system.
   - Required Python libraries: `pandas`, `scikit-learn`, `sqlite3`, `joblib`.
   - Place your dataset in the `data/` directory and the trained model in the `models/` directory.

2. **Running the Pipeline**:
   - Navigate to the project's root directory.
   - Run `python main.py` to start the ETL process and model predictions.
   - Run `python main.py predict` to start the prediction process using `predictor.py`.

3. **Running Tests**:
   - Execute `python -m unittest discover -s tests` to run the test cases.

## Contributing

Feel free to fork this project, make changes, and submit pull requests. Contributions to enhance the functionality or efficiency of this pipeline are always welcome.
