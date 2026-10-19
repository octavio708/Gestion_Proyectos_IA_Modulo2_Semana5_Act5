# Actividad 5: Detección de Fraude en Tarjetas de Crédito con MLflow

Este repositorio contiene la solución modular para la materia **Gestión de Proyectos de Inteligencia Artificial**. Implementa un pipeline completo de Machine Learning con prevención estricta de *data leakage*, búsqueda de hiperparámetros vía `GridSearchCV` e integración con **MLflow** para la trazabilidad.

## Estructura del Repositorio
- `datos/datos_ini/`: Archivo fuente `creditcard.csv`.
- `fuentes/datos_prep.py`: Pipeline de preprocesamiento con `ColumnTransformer` y `RobustScaler`.
- `fuentes/train.py`: Script CLI para entrenamiento, CV StratifiedKFold y logging en MLflow.
- `fuentes/entrena.ipynb`: Exploración visual (EDA).
- `CHANGELOG.md`: Registro de versión de artefactos y datasets.

## Instrucciones de Reproducción

1. **Clonar e instalar dependencias:**
   ```bash
   git clone <URL_REPOSITORIO>
   cd Actividad5
   pip install pandas numpy scikit-learn matplotlib mlflow xgboost