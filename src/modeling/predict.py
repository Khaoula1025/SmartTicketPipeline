import numpy as np
from src.modeling.train import load_data_and_embeddings, prepare_features_and_target,split_train_test, train_model,evaluate_model


def train_classification_pipeline(
    csv_path='tickets_cleaned.csv',
    embeddings_path='data/embeddings/tickets_embeddings.npy',
    model_type='logistic_regression',
    test_size=0.2):
    """
    Pipeline complet d'entraînement du modèle de classification
    
    Args:
        csv_path (str): Chemin vers le dataset nettoyé
        embeddings_path (str): Chemin vers les embeddings
        model_type (str): Type de modèle ('logistic_regression' ou 'random_forest')
        test_size (float): Proportion du test set
        output_dir (str): Dossier de sortie pour le modèle
    
    Returns:
        tuple: (model, metrics, X_test, y_test, y_pred)
    """
    # 1. Charger les données et embeddings
    df, embeddings = load_data_and_embeddings(csv_path, embeddings_path)
    
    # 2. Préparer X et y
    X, y, classes = prepare_features_and_target(df, embeddings)
    
    # 🔧 FIX: Convert y to numpy array to avoid PyArrow indexing issues
    if hasattr(y, 'values'):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # 3. Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size)
    
    # 4. Entraîner le modèle
    model = train_model(X_train, y_train, model_type)
    
    # 5. Évaluer le modèle
    metrics, y_pred = evaluate_model(model, X_test, y_test, classes)
    
    print("\n" + "="*70)
    print(" ✅ ÉTAPE 3 TERMINÉE ".center(70))
    print("="*70)
    print(f"\n📊 Résumé des performances:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   - Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"   - Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"   - F1-score (weighted): {metrics['f1_weighted']:.4f}")
    return model, metrics, X_test, y_test, y_pred


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    # Exécuter le pipeline complet
    model, metrics, X_test, y_test, y_pred = train_classification_pipeline(
        csv_path='data/processed/tickets_cleaned.csv',
        embeddings_path='data/embeddings/tickets_embeddings.npy',
        model_type='logistic_regression',  # ou 'random_forest'
        test_size=0.2,
        output_dir='models'
    )
