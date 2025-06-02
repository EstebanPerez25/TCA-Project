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
                # kedro run -p data_science -n reservaciones_exp1_namespace.split_data_node
            ),
            node(
                func=train_LogisticRegression,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=train_RandomForestClassifier,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="regressor_rf",
                name="train_rf_model_node",
            ),
            node(
                func=train_XGBClassifier,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="regressor_xgb",
                name="train_xgb_model_node",
            ),
            node(
                func=train_SupportVectorClassification,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="regressor_svc",
                name="train_svc_model_node",
            ),
            node(
                func=train_BalancedRandomForestClassifier,
                inputs=["X_train", "y_train", "params:model_params"],
                outputs="regressor_brf",
                name="train_brf_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
    ds_pipeline_exp1 = pipeline(
    pipe=pipeline_instance,
    inputs={"reservaciones_exp1": "reservaciones_exp1"},
    namespace="exp1",
    )

    ds_pipeline_exp2 = pipeline(
        pipe=pipeline_instance,
        inputs={"reservaciones_exp1": "reservaciones_exp2"},
        namespace="exp2",
    )

    return ds_pipeline_exp1 + ds_pipeline_exp2
