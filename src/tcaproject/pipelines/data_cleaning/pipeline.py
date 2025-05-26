"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import set_datatypes, drop_na


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=set_datatypes,
            inputs=["reservaciones", "params:datetime_columns", "params:category_columns", "params:bool_columns"],
            outputs="reservaciones_datatypes",
            name="set_datatypes_node",
        ),
        node(
            func=drop_na,
            inputs="reservaciones_datatypes",
            outputs="reservaciones_cleaned",
            name="drop_na_node",
        )
    ])
