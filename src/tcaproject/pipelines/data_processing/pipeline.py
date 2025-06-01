"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data, create_reservaciones_exp1


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["reservaciones", "params:datetime_columns", "params:category_columns", "params:bool_columns"],
            outputs="reservaciones_cleaned",
            name="clean_node",
        ),
        node(
            func=create_reservaciones_exp1,
            inputs=["reservaciones_cleaned", "params:drop_columns", "params:target_encoding_columns"],
            outputs="reservaciones_exp1",
            name="create_features_node"
        )
    ])
