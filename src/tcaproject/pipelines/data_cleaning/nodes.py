"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.13
"""
import pandas as pd


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


