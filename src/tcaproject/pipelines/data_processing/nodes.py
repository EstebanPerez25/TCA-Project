"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.13
"""
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import (  # noqa: F401
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


# 02_intermediate
def _change_datatype_to_datetime(x:pd.Series) -> pd.Series:
    return x.astype('datetime64[ns]')

def _change_datatype_to_object(x:pd.Series) -> pd.Series:
    return x.astype('object')

def _change_datatype_to_bool(x:pd.Series) -> pd.Series:
    return x.astype('bool')

def _capitalize_strings(x:pd.Series) -> pd.Series:
    #return x.str.capitalize()
    return x.str.title()

def _drop_nas(x:pd.DataFrame) -> pd.DataFrame:
    return x.dropna(axis=0)

def _clean_spaces(x:pd.DataFrame) -> pd.DataFrame:
    # Clean extra scpaces between text, before and after text
    return x.map(lambda s: ' '.join(s.split()) if isinstance(s, str) else s)

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
    df = _clean_spaces(df)
    return df

# 03_primary
def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df["cancelacion"] = df["nombre_estatus_reservacion"].apply(lambda x: True if x == 'Reservacion Cancelada' else False)
    return df

def _create_days_in_advance_variable(df: pd.DataFrame) -> pd.DataFrame:
    df['dias_anticipacion'] = (df['fecha_llegada'] - df['fecha_reservacion']).dt.days
    return df

def _drop_variables(df:pd.DataFrame, drop_variables:list[str]) -> pd.DataFrame:
    return df.drop([x for x in drop_variables], axis=1)

def _convert_variables_to_target_encoding(df:pd.DataFrame, target_enc_variables:list[str]) -> pd.DataFrame:
    for var in target_enc_variables:
        encoder = TargetEncoder(cols=[var], smoothing=3)
        encoder.fit(df[var], df['cancelacion'])
        df[var + '_te'] = encoder.transform(df[var])
        df.drop(var, axis=1, inplace=True)
    return df

def _scale_numeric_variables(df: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df.copy()
    scale_variables = df.select_dtypes(include=['int64', 'float64']).columns

    df_scaled[scale_variables] = RobustScaler().fit_transform(df[scale_variables])
    return df_scaled

def create_reservaciones_exp1(df: pd.DataFrame, drop_variables, target_enc_variables) -> pd.DataFrame:
    df = _create_target_variable(df)
    df = _create_days_in_advance_variable(df)
    df = _drop_variables(df, drop_variables)
    df = _convert_variables_to_target_encoding(df, target_enc_variables)
    df = _scale_numeric_variables(df)
    return df
