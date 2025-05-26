"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.13
"""
import pandas as pd


# 02_intermediate
def _change_datatype_to_datetime(x:pd.Series) -> pd.Series:
    return x.astype('datetime64[ns]')

def _change_datatype_to_object(x:pd.Series) -> pd.Series:
    return x.astype('object')

def _change_datatype_to_bool(x:pd.Series) -> pd.Series:
    return x.astype('bool')

def _capitalize_strings(x:pd.Series) -> pd.Series:
    return x.str.capitalize()

def _drop_nas(x:pd.DataFrame) -> pd.DataFrame:
    return x.dropna(axis=0)

def clean_data(
    df: pd.DataFrame,
    datetime_columns: list[str],
    category_columns: list[str],
    bool_columns: list[str]
) -> pd.DataFrame:
    for col in datetime_columns:
        df[col] = _change_datatype_to_datetime(df[col])
    for col in category_columns:
        df[col] = _change_datatype_to_object(df[col])
        df[col] = _capitalize_strings(df[col])
    for col in bool_columns:
        df[col] = _change_datatype_to_bool(df[col])

    df = _drop_nas(df)
    return df

# 03_primary
def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df["cancelacion"] = df["nombre_estatus_reservacion"].apply(lambda x: True if x == 'Reservacion Cancelada' else False)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _create_target_variable(df)
    return df
