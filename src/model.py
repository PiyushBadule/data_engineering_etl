import joblib
from sklearn.ensemble import RandomForestRegressor

class ModelHandler:
    """
    Class to handle machine learning model operations.
    """

    def __init__(self):
        """
        Initialize ModelHandler object.
        """
        self.model = None

    def train(self, X_train, y_train):
        """
        Train a RandomForestRegressor model on the provided training data.

        :param X_train: Features of the training data.
        :param y_train: Target variable of the training data.
        :return: Trained RandomForestRegressor model.
        """
        try:
            regr = RandomForestRegressor(max_depth=12)
            regr.fit(X_train, y_train)
            return regr
        except Exception as e:
            raise ValueError(f"Training failed: {e}")

    def predict(self, X, model):
        """
        Make predictions using the provided model on the given data.

        :param X: Data on which predictions are to be made.
        :param model: Trained machine learning model.
        :return: Predictions made by the model.
        """
        try:
            return model.predict(X)
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

    def save_model(self, model, filename):
        """
        Save the given machine learning model to a file.

        :param model: Machine learning model to be saved.
        :param filename: Path where the model will be saved.
        """
        try:
            with open(filename, 'wb') as file:
                joblib.dump(model, file, compress=3)
        except Exception as e:
            raise IOError(f"Failed to save model: {e}")

    def load_model(self, filename):
        """
        Load a machine learning model from the specified file.

        :param filename: Path to the file containing the saved model.
        :return: Loaded machine learning model.
        """
        try:
            return joblib.load(filename)
        except Exception as e:
            raise IOError(f"Failed to load model: {e}")
