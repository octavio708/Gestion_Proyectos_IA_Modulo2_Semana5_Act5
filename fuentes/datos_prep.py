"""
datos_prep.py - Preprocesamiento y pipeline modular de datos.
Garantiza cero data leakage al encapsular transformaciones en ColumnTransformer.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


def crear_pipeline_preprocesamiento():
    """
    Crea un ColumnTransformer que aplica RobustScaler a 'Time' y 'Amount',
    mientras que deja intactas las variables PCA (V1-V28).
    """
    col_escalar = ['Time', 'Amount']
    
    preprocesador = ColumnTransformer(
        transformers=[
            ('scaler', RobustScaler(), col_escalar)
        ],
        remainder='passthrough'
    )
    return preprocesador


def obtener_pipeline_completo(modelo):
    """
    Ensambla el preprocesador y el modelo de ML en un único Pipeline de Scikit-Learn.
    """
    preprocesador = crear_pipeline_preprocesamiento()
    pipeline = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('clasificador', modelo)
    ])
    return pipeline