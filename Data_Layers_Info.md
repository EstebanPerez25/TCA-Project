# Data Layers in Data Engineering

https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71/


### Summary

- **01_raw**  
  Contiene los datos originales sin modificar, tal como fueron obtenidos de la fuente. Este layer es la única fuente de verdad y no debe alterarse. Los datos pueden estar en formatos sin tipar, como CSV o JSON.  
  *Ejemplo:* Archivos CSV con registros de ventas sin limpieza ni transformación.

- **02_intermediate**  
  (Opcional) Representa los datos del layer raw pero con tipos definidos y limpieza básica, como conversión de cadenas a números o fechas, y corrección de nombres de campos. No se cambia la estructura original del dato.  
  *Ejemplo:* Mismo CSV convertido a formato Parquet con columnas numéricas y de fecha correctamente tipadas.

- **03_primary**  
  Modelo de datos específico del dominio, con datos limpios y transformados para que encajen con el problema que se quiere resolver. Es el espacio de trabajo para ingeniería de características.  
  *Ejemplo:* Tabla con ventas agregadas por cliente y fecha, lista para análisis o modelado.

- **04_feature**  
  Modelos de datos analíticos que contienen conjuntos de características (features) derivadas del layer primary, agrupadas por área de análisis y ligadas a dimensiones comunes. Son las variables independientes y objetivo para ML.  
  *Ejemplo:* Dataset con variables como edad, ingreso, y etiqueta de compra para entrenamiento de modelos.

- **05_model_input**  
  Modelos analíticos que consolidan todas las features contra una dimensión común y, en proyectos en vivo, incluyen fecha de ejecución para versionar el histórico. También llamados "Master Tables".  
  *Ejemplo:* Tabla maestra con todas las características y etiquetas para un periodo específico, lista para alimentar modelos.

- **06_models**  
  Modelos de machine learning entrenados y serializados, almacenados para su uso posterior. Pueden ser archivos pickle o gestionados con frameworks MLOps como MLFlow.  
  *Ejemplo:* Archivo `.pkl` con un modelo de regresión entrenado.

- **07_model output**  
  Resultados generados por los modelos a partir de los datos de entrada, como predicciones o scores.  
  *Ejemplo:* Tabla con predicciones de ventas futuras por cliente.

- **08_reporting**  
  Informes o análisis descriptivos, a menudo ad hoc, que muestran resultados de modelos o análisis para toma de decisiones.  
  *Ejemplo:* Dashboard con gráficos de desempeño del modelo y métricas de negocio.

Este esquema permite organizar el flujo de datos desde la adquisición hasta la generación de valor analítico y de negocio, facilitando la trazabilidad, reproducibilidad y escalabilidad en proyectos de ciencia de datos y MLOps.





### Full Table

| Layer  |Order| Description  |
|- |-|- |
| `raw`  |Sequential| **Initial start of the pipeline**, containing the *sourced data model(s) that should never be changed*, it forms your single source of truth to work from. These data models can be un-typed in most cases e.g. `csv`, but this will vary from case to case. Given the relative cost of storage today, painful experience suggests it's safer to never work with the original data directly! |
| `intermediate`  |Sequential| **This stage is optional if your data is already typed.** Typed representation of the raw layer e.g. converting string based values into their current typed representation as numbers, dates etc. Our recommended approach is to mirror the raw layer in a typed format like Apache [Parquet](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.pandas.ParquetDataSet.html). Avoid transforming the structure of the data, but simple operations like cleaning up field names or 'unioning' mutli-part CSVs are permitted.  |
| `primary`  |Sequential| **Domain specific data model(s)** containing cleansed, transformed and wrangled data from either raw or intermediate, which forms your layer that can be treated as the workspace for any feature engineering down the line. This holds the data transformed into a model that fits the problem domain in question. If you are working with data which is already formatted for the problem domain, it is reasonable to skip to this point.|
| `feature`  |Sequential| **Analytics specific data model(s)** containing a set of features defined against the primary data, which are grouped by feature area of analysis and stored against a common dimension. In practice this covers the independent variables and target variable which will form the basis for ML exploration and application. Since this framework was designed MLOps tooling has progressed and now 'Feature Stores' (such as [Feast](https://feast.dev/) or [SageMaker Feature Store](https://aws.amazon.com/sagemaker/feature-store/)) provide a versioned, centralised storage location with low-latency serving. This separation still fits in well within this conceptual framework.|
| `model_input`  |Sequential| **Analytics specific data model(s)** containing all feature data against a common dimension and in the case of live projects against an analytics run date to ensure that you track the historical changes of the features over time. Many places call these the 'Master Table(s)', we believe this terminology is more precise and covers multi-models pipelines better.|
| `models`  |Sequential| **Stored, serialised pre-trained machine learning models**. In the simplest case, these are stored as something like a [pickle](https://docs.python.org/3/library/pickle.html) file on a filesystem. More mature implementations would leverage MLOps frameworks that provide model serving such as [MLFlow](https://databricks.com/blog/2020/06/25/announcing-mlflow-model-serving-on-databricks.html).   |
| `model output`  |Sequential| Analytics specific data model(s) containing the results generated by the model based on the model input data.  
|`reporting`|Free-form| Used for outputting analyses or modeling results that are often Ad Hoc or simply descriptive reports.