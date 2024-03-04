import pandas as pd
from sklearn.model_selection import train_test_split
from constants import RANDOM_STATE, TEST_SIZE, FILTERED_COLUMNS, DROP_COLUMN, COLUMN_MAPPING
from src.database import DatabaseHandler

class DataProcessor:
    """
    Class for processing housing data and preparing it for training.
    """

    def prepare_data(self, input_data_path):
        """
        Processes the housing data from a given CSV file and saves the transformed data to the database.
        The function also performs a train-test split on the data.

        :param input_data_path: Path to the CSV file containing housing data.
        :return: Tuple containing split data (X_train, X_test, y_train, y_test).
        """
        try:
            # Read the data from CSV
            df = pd.read_csv(input_data_path)
            # Drop rows with NaN values
            df = df.dropna()

            # Drop rows with 'Null' values in categorical columns
            for col in df.columns[df.dtypes == 'O']:
                df.drop(df[df[col] == 'Null'].index, inplace=True)

            # Separate features and target variable
            df_features = df.drop(DROP_COLUMN, axis=1)
            y = df['MEDIAN_HOUSE_VALUE'].values

            # One-hot encode categorical features
            df_features = pd.get_dummies(df_features, columns=['OCEAN_PROXIMITY'])

            # Filter relevant columns
            filtered_columns = FILTERED_COLUMNS
            df_features = df_features.loc[:, filtered_columns]

            # Rename columns for consistency
            column_mapping = COLUMN_MAPPING
            df_features.rename(columns=column_mapping, inplace=True)

            # Save transformed data to the database
            DatabaseHandler().save_to_database(df_features)

            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=TEST_SIZE,
                                                                random_state=RANDOM_STATE)
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise Exception(f"Error in data preparation: {e}")
