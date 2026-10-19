"""
train.py - Entrenamiento, ajuste de hiperparámetros y registro en MLflow.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

from datos_prep import obtener_pipeline_completo


def cargar_datos(ruta_csv='../datos/datos_ini/creditcard.csv'):
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    X = df.drop(columns=['Class'])
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluar_y_registrar(nombre_experimento, modelo_base, param_grid, X_train, X_test, y_train, y_test):
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    with mlflow.start_run(run_name=nombre_experimento):
        pipeline = obtener_pipeline_completo(modelo_base)
        
        # Cross-Validation StratifiedKFold para manejar desbalance sin leakage
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Ajuste del Grid Search enfocado en Recall (minimizar Falsos Negativos)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='recall',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred = mejor_modelo.predict(X_test)
        y_proba = mejor_modelo.predict_proba(X_test)[:, 1]
        
        # Cálculo de Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Registrar Hiperparámetros y Métricas en MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Generar y guardar Artefacto: Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - {nombre_experimento}")
        fig_path = f"{nombre_experimento}_cm.png"
        plt.savefig(fig_path)
        plt.close()
        
        mlflow.log_artifact(fig_path)
        os.remove(fig_path)
        
        # Registrar Modelo
        mlflow.sklearn.log_model(mejor_modelo, artifact_path="model")
        
        print(f"[{nombre_experimento}] Completado. Mejor Recall CV: {grid_search.best_score_:.4f} | Test Recall: {rec:.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_datos()
    
    # 1. Regresión Logística (Modelo Base)
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
    
    # 2. Random Forest (Modelo Ensamblado)
    grid_rf = {
        'clasificador__n_estimators': [50, 100],
        'clasificador__max_depth': [10, 20],
        'clasificador__class_weight': ['balanced_subsample']
    }
    evaluar_y_registrar(
        "Random_Forest_Complex",
        RandomForestClassifier(random_state=42),
        grid_rf, X_train, X_test, y_train, y_test
    )