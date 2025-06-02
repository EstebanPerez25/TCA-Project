"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.13
"""

import logging

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def split_data(df:pd.DataFrame, parameters: dict) -> tuple:  # noqa: UP006
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters_data_science.yml.
    Returns:
        Split data.
    """
    X = df.drop("cancelacion", axis=1)
    y = df["cancelacion"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"], random_state=parameters["random_state"])
    return X_train, X_test, y_train, y_test.to_frame(name="cancelacion")

# Models Training
def train_LogisticRegression(X_train: pd.DataFrame, y_train: pd.Series, hiperparameters: dict) -> LogisticRegression:
    """Trains the Logistic Regression model.
    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
    Returns:
        Trained model.
    """
    model = LogisticRegression(
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
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
    model = RandomForestClassifier(
        class_weight='balanced'
        )

    model.fit(X_train, y_train)
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
    model = XGBClassifier(
        scale_pos_weight=hiperparameters["scale_pos_weight"]
    )

    model.fit(X_train, y_train)
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
    model = SVC(
        #probability=True, # Time consuming
        class_weight='balanced'
        )

    model.fit(X_train, y_train)
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
    model = BalancedRandomForestClassifier(
        sampling_strategy='auto',
        replacement=True
        )

    model.fit(X_train, y_train)
    logger.info("Balanced Random Forest Classifier trained successfully.")
    return model


# Evaluation
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    if hasattr(model, "predict_proba"):
        ap = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        ap = float("nan")
    report = classification_report(y_test, y_pred, target_names=['No Cancellation', 'Cancellation'])

    logger.info("Model evaluation metrics:")
    logger.info("Precision: %.3f \t Recall: %.3f \t F1 Score: %.3f", precision, recall, f1)
    logger.info("Average Precision: %.3f", ap)
    logger.info("Classification Report:\n%s", report)

    # Return metrics as a dictionary
    return {
        "precision": {"value": precision},
        "recall": {"value": recall},
        "f1_score": {"value": f1},
        "average_precision": {"value": ap}
    }
