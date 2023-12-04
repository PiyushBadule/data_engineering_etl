import pandas as pd
from sklearn.model_selection import train_test_split

from data_engineering_etl.src.database import save_to_database

RANDOM_STATE = 100


def prepare_data(input_data_path, conn):
    """
    Processes the housing data from a given CSV file and saves the transformed data to the database.
    The function also performs a train-test split on the data.

    :param input_data_path: Path to the CSV file containing housing data.
    :param conn: Database connection object.
    :return: Tuple containing split data (X_train, X_test, y_train, y_test).
    """
    try:
        df = pd.read_csv(input_data_path)
        df = df.dropna()

        for col in df.columns[df.dtypes == 'O']:
            df.drop(df[df[col] == 'Null'].index, inplace=True)

        df_features = df.drop(['MEDIAN_HOUSE_VALUE', 'AGENCY'], axis=1)
        y = df['MEDIAN_HOUSE_VALUE'].values

        df_features = pd.get_dummies(df_features, columns=['OCEAN_PROXIMITY'])

        filtered_columns = ['LONGITUDE', 'LAT', 'MEDIAN_AGE', 'ROOMS', 'BEDROOMS', 'POP',
                            'HOUSEHOLDS', 'MEDIAN_INCOME', 'OCEAN_PROXIMITY_<1H OCEAN',
                            'OCEAN_PROXIMITY_INLAND', 'OCEAN_PROXIMITY_ISLAND',
                            'OCEAN_PROXIMITY_NEAR BAY', 'OCEAN_PROXIMITY_NEAR OCEAN']
        df_features = df_features.loc[:, filtered_columns]

        column_mapping = {'LONGITUDE': 'longitude',
                          'LAT': 'latitude',
                          'MEDIAN_AGE': 'housing_median_age',
                          'ROOMS': 'total_rooms',
                          'BEDROOMS': 'total_bedrooms',
                          'POP': 'population',
                          'HOUSEHOLDS': 'households',
                          'MEDIAN_INCOME': 'median_income',
                          'OCEAN_PROXIMITY_<1H OCEAN': 'ocean_proximity_<1H OCEAN',
                          'OCEAN_PROXIMITY_INLAND': 'ocean_proximity_INLAND',
                          'OCEAN_PROXIMITY_ISLAND': 'ocean_proximity_ISLAND',
                          'OCEAN_PROXIMITY_NEAR BAY': 'ocean_proximity_NEAR BAY',
                          'OCEAN_PROXIMITY_NEAR OCEAN': 'ocean_proximity_NEAR OCEAN'}
        df_features.rename(columns=column_mapping, inplace=True)

        save_to_database(df_features, conn)

        X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=RANDOM_STATE)
        return (X_train, X_test, y_train, y_test)

    except Exception as e:
        raise Exception(f"Error in data preparation: {e}")
