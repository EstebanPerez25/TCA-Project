"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["reservaciones", "params:datetime_columns", "params:category_columns", "params:bool_columns"],
            outputs="reservaciones_cleaned",
            name="clean_node",
        )
    ])
