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

def _clean_accents(x:pd.Series) -> pd.Series:
    # Clean accents in strings
    import unicodedata
    return x.apply(lambda s: unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8') if isinstance(s, str) else s)

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
        df[col] = _clean_accents(df[col])
    for col in bool_columns:
        df[col] = _change_datatype_to_bool(df[col])

    df = _drop_nas(df)
    df = _clean_spaces(df)
    return df

# 03_primary
def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df["cancelacion"] = df["nombre_estatus_reservacion"].apply(lambda x: 1 if x == 'Reservacion Cancelada' else 0)
    return df

def _create_days_in_advance_variable(df: pd.DataFrame) -> pd.DataFrame:
    df['dias_llegada-reservacion'] = (df['fecha_llegada'] - df['fecha_reservacion']).dt.days
    df['dias_salida-llegada'] = (df['fecha_salida'] - df['fecha_llegada']).dt.days
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

def _change_bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    return df

def _scale_numeric_variables(df: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df.copy()
    scale_variables = df.select_dtypes(include=['int64', 'float64']).columns

    df_scaled[scale_variables] = RobustScaler().fit_transform(df[scale_variables])
    return df_scaled

def _filter_model_variables(df: pd.DataFrame, model_variables: list[str]) -> None:
    #for experiment 2, we only keep the model variables and target variable
    return df[model_variables + ['cancelacion']]

# pesos_variables
def calcular_peso_columna(dfx, columna, nombre_columna_resultado=None):
    nombre_columna_resultado = nombre_columna_resultado or columna
    conteos = dfx[columna].astype(str).value_counts()
    peso = conteos.rename_axis(nombre_columna_resultado).reset_index(name='conteo')
    peso['proporcion'] = (peso['conteo'] / peso['conteo'].sum()).round(4)
    return peso

def create_pesos_variables_json(df: pd.DataFrame, model_variables: list[str]) -> dict:
    df = _create_target_variable(df)
    df = _filter_model_variables(df, model_variables)
    df_cancelations = df[df['cancelacion'] == 1]
    pesos = {
    f"peso_{col}": calcular_peso_columna(dfx=df_cancelations, columna=col)
    for col in model_variables
}
    dict_pesos = {k: v.to_dict(orient='records') for k, v in pesos.items()}
    return dict_pesos

def create_reservaciones_exp1(df: pd.DataFrame, drop_variables, target_enc_variables) -> pd.DataFrame:
    df = _create_target_variable(df)
    df = _create_days_in_advance_variable(df)
    df = _drop_variables(df, drop_variables)
    df = _convert_variables_to_target_encoding(df, target_enc_variables)
    df = _change_bool_to_int(df)
    df = _scale_numeric_variables(df)
    return df

def create_reservaciones_exp2(df: pd.DataFrame, model_variables) -> pd.DataFrame:
    df = _create_target_variable(df)
    df = _filter_model_variables(df, model_variables)
    #df = _convert_variables_to_target_encoding(df, target_enc_variables)
    return df
