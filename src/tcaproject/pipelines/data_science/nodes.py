"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.13
"""

import logging

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def split_data(data:pd.DataFrame, parameters: dict) -> tuple:  # noqa: UP006
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters_data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])
    return X_train, X_test, y_train, y_test

# Models Training
def train_LogisticRegression(X_train: pd.DataFrame, y_train: pd.Series, hiperparameters: dict) -> LogisticRegression:
    """Trains the Logistic Regression model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train,
              class_weight='balanced'
              )

    logger.info("Logistic Regression trained successfully.")
    return model

def train_RandomForestClassifier(X_train: pd.DataFrame, y_train: pd.Series, hiperparameters: dict) -> RandomForestClassifier:
    """Trains the Random Forest Classifier model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train,
              class_weight='balanced'
              )
    logger.info("Random Forest Classifier trained successfully.")
    return model

def train_XGBClassifier(X_train: pd.DataFrame, y_train: pd.Series, hiperparameters: dict) -> XGBClassifier:
    """Trains the SGBoost Classifier model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    model = XGBClassifier()
    model.fit(X_train, y_train,
              scale_pos_weight=hiperparameters["scale_pos_weight"]
              )
    logger.info("XGBoost Classifier trained successfully.")
    return model

def train_SupportVectorClassification(X_train: pd.DataFrame, y_train: pd.Series, hiperparameters: dict) -> SVC:
    """Trains the Support Vector Classification model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    model = SVC()
    model.fit(X_train, y_train,
              class_weight='balanced'
              )
    logger.info("Support Vector Classification trained successfully.")
    return model

def train_BalancedRandomForestClassifier(X_train: pd.DataFrame, y_train: pd.Series, hiperparameters: dict) -> BalancedRandomForestClassifier:
    """Trains the Balanced Random Forest Classifier model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    model = BalancedRandomForestClassifier()
    model.fit(X_train, y_train,
              sampling_strategy='auto',
              replacement=True
              )
    logger.info("Balanced Random Forest Classifier trained successfully.")
    return model


# Evaluation
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = model.predict(X_test)
    # r2 = r2_score(y_test, y_pred)
    # r2_score_adjusted = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    # rmse = root_mean_squared_error(y_test, y_pred)
    # mape = mean_absolute_percentage_error(y_test, y_pred)
    # logger.info("R^2: %.3f \t Adjusted R^2: %.3f \t RMSE: %.3f \t MAPE: %.3f", r2, r2_score_adjusted, rmse, mape)
    pass