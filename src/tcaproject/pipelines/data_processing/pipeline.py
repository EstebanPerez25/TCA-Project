"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data, create_reservaciones_exp1, create_pesos_variables_json, create_reservaciones_exp2


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
            name="create_reservaciones_exp1_node"
        ),
        node(
            func=create_pesos_variables_json,
            inputs=["reservaciones_cleaned", "params:model_variables_exp2"],
            outputs="pesos_variables_json",
            name="create_pesos_variables_json_node"
        ),
        node(
            func=create_reservaciones_exp2,
            inputs=["reservaciones_cleaned", "params:model_variables_exp2"],
            outputs="reservaciones_exp2",
            name="create_reservaciones_exp2_node"
        )
    ])
