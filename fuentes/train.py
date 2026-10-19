"""
train.py - Entrenamiento, optimización de hiperparámetros y registro en MLflow.
Optimizado con soporte CUDA para GPU NVIDIA (RTX 4070 Ti) mediante XGBoost.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from datos_prep import obtener_pipeline_completo


def cargar_datos(ruta_csv='../datos/datos_ini/creditcard.csv'):
    """
    Carga el dataset creditcard.csv y realiza la división estratificada
    para mantener la proporción de la clase minoritaria (fraudes).
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo en {ruta_csv}. "
            f"Asegúrate de colocar 'creditcard.csv' en la carpeta datos/datos_ini/"
        )
    
    df = pd.read_csv(ruta_csv)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # División estratificada 80/20
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluar_y_registrar(nombre_experimento, modelo_base, param_grid, X_train, X_test, y_train, y_test):
    """
    Ejecuta Grid Search CV, predice sobre el conjunto de prueba y registra
    parámetros, métricas y artefactos visuales en MLflow.
    """
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    with mlflow.start_run(run_name=nombre_experimento):
        # Ensamblar Pipeline con el preprocesador modular
        pipeline = obtener_pipeline_completo(modelo_base)
        
        # Cross-Validation StratifiedKFold (5 splits) para evitar leakage
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Búsqueda de hiperparámetros optimizando Recall (minimizar Falsos Negativos)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='recall',
            n_jobs=1
        )
        
        print(f"\nIniciando entrenamiento para: {nombre_experimento}...")
        grid_search.fit(X_train, y_train)
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones sobre el conjunto de prueba (Test set independiente)
        y_pred = mejor_modelo.predict(X_test)
        y_proba = mejor_modelo.predict_proba(X_test)[:, 1]
        
        # Cálculo de Métricas Evaluativas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Log de Parámetros y Métricas en MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Generar y Registrar Artefacto Visual: Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legítimo', 'Fraude'])
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title(f"Matriz de Confusión - {nombre_experimento}")
        
        fig_path = f"{nombre_experimento}_cm.png"
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(fig_path)
        if os.path.exists(fig_path):
            os.remove(fig_path)
            
        # Log del Modelo Serializado
        mlflow.sklearn.log_model(
        mejor_modelo, 
        artifact_path="model",
        skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']
        )
        
        print(f"[{nombre_experimento}] ¡Completado exitosamente!")
        print(f"  - Mejor Recall CV:  {grid_search.best_score_:.4f}")
        print(f"  - Test Recall:     {rec:.4f}")
        print(f"  - Test Precision:  {prec:.4f}")
        print(f"  - Test F1-Score:   {f1:.4f}")
        print(f"  - Test ROC-AUC:    {roc_auc:.4f}")


if __name__ == "__main__":
    # Cargar y dividir los datos
    X_train, X_test, y_train, y_test = cargar_datos()
    
    # ---------------------------------------------------------
    # 1. Regresión Logística (Modelo Base / CPU)
    # ---------------------------------------------------------
    grid_lr = {
        'clasificador__C': [0.01, 0.1, 1.0, 10.0],
        'clasificador__class_weight': ['balanced'],
        'clasificador__max_iter': [1000]
    }
    
    evaluar_y_registrar(
        "Logistic_Regression_Baseline",
        LogisticRegression(random_state=42),
        grid_lr, X_train, X_test, y_train, y_test
    )
    
    # ---------------------------------------------------------
    # 2. XGBoost Classifier (Modelo Complejo / GPU CUDA RTX 4070 Ti)
    # ---------------------------------------------------------
    grid_xgb = {
        'clasificador__n_estimators': [100],        # Una sola opción para probar rápido
        'clasificador__max_depth': [6],              # Profundidad moderada
        'clasificador__learning_rate': [0.1],
        'clasificador__scale_pos_weight': [577]
    }
    
    evaluar_y_registrar(
        "XGBoost_GPU_Optimized",
        XGBClassifier(
            device='cuda',        # Aceleración por hardware en tu RTX 4070 Ti
            tree_method='hist',   # Algoritmo por histogramas optimizado para GPU
            random_state=42,
            eval_metric='logloss'
        ),
        grid_xgb, X_train, X_test, y_train, y_test
    )