import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)

def evaluate_model(model, X_test, y_test, classes):
    """
    Évaluer le modèle sur le test set
    
    Args:
        model: Modèle entraîné
        X_test (np.ndarray): Features de test
        y_test (np.ndarray): Target de test
        classes (np.ndarray): Liste des classes
    
    Returns:
        dict: Métriques d'évaluation
    """
    print(f"\n📊 Évaluation du modèle...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Precision, Recall, F1 par classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=classes
    )
    
    # Métriques globales (weighted average)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Stocker les métriques
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision_avg,
        'recall_weighted': recall_avg,
        'f1_weighted': f1_avg,
        'precision_per_class': dict(zip(classes, precision)),
        'recall_per_class': dict(zip(classes, recall)),
        'f1_per_class': dict(zip(classes, f1)),
        'support_per_class': dict(zip(classes, support))
    }
    
    return metrics, y_pred
