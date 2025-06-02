"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (split_data,
                    train_LogisticRegression,
                    train_RandomForestClassifier,
                    train_XGBClassifier,
                    train_SupportVectorClassification,
                    train_BalancedRandomForestClassifier,
                    evaluate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=split_data,
                inputs=["reservaciones_exp1", "params:split_data_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
                # kedro run -p data_science -n exp1.split_data_node
            ),

            # Logistic Regression
            node(
                func=train_LogisticRegression,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="logistic_regressor",
                name="train_lr_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["logistic_regressor", "X_test", "y_test"],
                outputs="lr_model_metrics",
                name="evaluate_lr_model_node",
            ),

            # Random Forest Classifier
            node(
                func=train_RandomForestClassifier,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="random_forest_classifier",
                name="train_rf_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["random_forest_classifier", "X_test", "y_test"],
                outputs="rf_model_metrics",
                name="evaluate_rf_model_node",
            ),

            # XGBoost Classifier
            node(
                func=train_XGBClassifier,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="xgboost_classifier",
                name="train_xgb_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["xgboost_classifier", "X_test", "y_test"],
                outputs="xgb_model_metrics",
                name="evaluate_xgb_model_node",
            ),

            # Support Vector Classification
            # node(
            #     func=train_SupportVectorClassification,
            #     inputs=["X_train", "y_train", "params:model_params"],
            #     outputs="support_vector_classifier",
            #     name="train_svc_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["support_vector_classifier", "X_test", "y_test"],
            #     outputs="svc_model_metrics",
            #     name="evaluate_svc_model_node",
            # ),

            # Balanced Random Forest Classifier
            node(
                func=train_BalancedRandomForestClassifier,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="balanced_random_forest_classifier",
                name="train_brf_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["balanced_random_forest_classifier", "X_test", "y_test"],
                outputs="brf_model_metrics",
                name="evaluate_brf_model_node",
            ),
        ]
    )
    ds_pipeline_exp1 = pipeline(
    pipe=pipeline_instance,
    inputs={"reservaciones_exp1": "reservaciones_exp1"},
    parameters={"split_data_params", "model_params"},
    namespace="exp1",
    )

    ds_pipeline_exp2 = pipeline(
        pipe=pipeline_instance,
        inputs={"reservaciones_exp1": "reservaciones_exp2"},
        parameters={"split_data_params", "model_params"},
        namespace="exp2",
    )

    return ds_pipeline_exp1 + ds_pipeline_exp2
