import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ============================================================
# FONCTIONS
# ============================================================

def load_data_and_embeddings(csv_path, embeddings_path):
    """
    Charger le dataset et les embeddings
    
    Args:
        csv_path (str): Chemin vers le CSV nettoyé
        embeddings_path (str): Chemin vers les embeddings (.npy)
    
    Returns:
        tuple: (df, embeddings)
    """
    print(f"\n Chargement des données...")
    
    # Charger le dataset
    df = pd.read_csv(csv_path)
    print(f" Dataset: {len(df)} lignes")
    
    # Charger les embeddings
    embeddings = np.load(embeddings_path)
    print(f" Embeddings: {embeddings.shape}")
    
    # Vérifier la cohérence
    if len(df) != len(embeddings):
        raise ValueError(
            f"Erreur: tailles différentes! "
            f"Dataset: {len(df)}, Embeddings: {len(embeddings)}"
        )
    
    print(f"✅ Cohérence vérifiée")
    
    return df, embeddings


def prepare_features_and_target(df, embeddings, target_column='type'):
    """
    Préparer les features (X) et la cible (y)
    
    Args:
        df (pd.DataFrame): Dataset
        embeddings (np.ndarray): Matrice d'embeddings
        target_column (str): Nom de la colonne cible
    
    Returns:
        tuple: (X, y, classes)
    """
    print(f"\n📊 Préparation des features et target...")
    
    # Features = embeddings
    X = embeddings
    
    # Target = type de ticket
    y = df[target_column].values
    
    # Classes uniques
    classes = np.unique(y)
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"✅ Nombre de classes: {len(classes)}")
    print(f"   Classes: {classes}")
    
    # Distribution des classes
    print(f"\n📊 Distribution des classes:")
    class_counts = pd.Series(y).value_counts()
    for cls, count in class_counts.items():
        percentage = (count / len(y)) * 100
        print(f"   {cls}: {count} ({percentage:.1f}%)")
    
    return X, y, classes


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Séparer les données en train et test
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target
        test_size (float): Proportion du test set
        random_state (int): Seed pour reproductibilité
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\n✂️  Séparation train/test...")
    print(f"   Test size: {test_size*100:.0f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Garde la même distribution de classes
    )
    
    print(f"✅ Split terminé:")
    print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Vérifier la distribution dans train et test
    print(f"\n📊 Distribution dans le train set:")
    train_dist = pd.Series(y_train).value_counts(normalize=True) * 100
    for cls, pct in train_dist.items():
        print(f"   {cls}: {pct:.1f}%")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type='logistic_regression'):
    """
    Entraîner le modèle de classification
    
    Args:
        X_train (np.ndarray): Features d'entraînement
        y_train (np.ndarray): Target d'entraînement
        model_type (str): Type de modèle ('logistic_regression' ou 'random_forest')
    
    Returns:
        model: Modèle entraîné
    """
    print(f"\n🤖 Entraînement du modèle: {model_type}...")
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,  # Utiliser tous les CPU
            verbose=0
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    # Entraîner
    model.fit(X_train, y_train)
    
    print(f"✅ Entraînement terminé")
    
    return model


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


def plot_confusion_matrix(y_test, y_pred, classes, output_path='confusion_matrix.png'):
    """
    Créer et sauvegarder la matrice de confusion
    
    Args:
        y_test (np.ndarray): Vraies étiquettes
        y_pred (np.ndarray): Prédictions
        classes (np.ndarray): Liste des classes
        output_path (str): Chemin de sortie
    """
    print(f"\n📈 Génération de la matrice de confusion...")
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Créer la figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Nombre de prédictions'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Matrice de confusion sauvegardée: {output_path}")
    plt.close()


def save_model(model, metrics, output_dir='models'):
    """
    Sauvegarder le modèle et les métriques
    
    Args:
        model: Modèle entraîné
        metrics (dict): Métriques d'évaluation
        output_dir (str): Dossier de sortie
    
    Returns:
        tuple: (model_path, metrics_path)
    """
    print(f"\n💾 Sauvegarde du modèle...")
    
    # Créer le dossier si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(output_dir, 'model_classifier.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Modèle: {model_path}")
    
    # Sauvegarder les classes
    classes_path = os.path.join(output_dir, 'model_classes.npy')
    np.save(classes_path, model.classes_)
    print(f"✅ Classes: {classes_path}")
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("MÉTRIQUES DU MODÈLE\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (weighted): {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-score (weighted): {metrics['f1_weighted']:.4f}\n\n")
        f.write("Métriques par classe:\n")
        f.write("-"*50 + "\n")
        for cls in model.classes_:
            f.write(f"\n{cls}:\n")
            f.write(f"  Precision: {metrics['precision_per_class'][cls]:.4f}\n")
            f.write(f"  Recall: {metrics['recall_per_class'][cls]:.4f}\n")
            f.write(f"  F1-score: {metrics['f1_per_class'][cls]:.4f}\n")
            f.write(f"  Support: {metrics['support_per_class'][cls]}\n")
    
    print(f"✅ Métriques: {metrics_path}")
    
    return model_path, metrics_path


def predict_new_ticket(model, encoder_model, text):
    """
    Prédire le type d'un nouveau ticket
    
    Args:
        model: Modèle de classification entraîné
        encoder_model: Modèle SentenceTransformer pour générer l'embedding
        text (str): Texte du ticket
    
    Returns:
        tuple: (prediction, probabilities)
    """
    # Générer l'embedding
    embedding = encoder_model.encode([text])
    
    # Prédire
    prediction = model.predict(embedding)[0]
    probabilities = model.predict_proba(embedding)[0]
    
    return prediction, dict(zip(model.classes_, probabilities))


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def train_classification_pipeline(
    csv_path='tickets_cleaned.csv',
    embeddings_path='data/embeddings/tickets_embeddings.npy',
    model_type='logistic_regression',
    test_size=0.2,
    output_dir='models'
):
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
    print("="*70)
    print(" ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION ".center(70))
    print("="*70)
    
    # 1. Charger les données et embeddings
    df, embeddings = load_data_and_embeddings(csv_path, embeddings_path)
    
    # 2. Préparer X et y
    X, y, classes = prepare_features_and_target(df, embeddings)
    
    # 3. Split train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size)
    
    # 4. Entraîner le modèle
    model = train_model(X_train, y_train, model_type)
    
    # 5. Évaluer le modèle
    metrics, y_pred = evaluate_model(model, X_test, y_test, classes)
    
    # 6. Visualiser la confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes, 'confusion_matrix.png')
    
    # 7. Sauvegarder le modèle
    save_model(model, metrics, output_dir)
    
    print("\n" + "="*70)
    print(" ✅ ÉTAPE 3 TERMINÉE ".center(70))
    print("="*70)
    print(f"\n📊 Résumé des performances:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   - Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"   - Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"   - F1-score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"\n📁 Fichiers générés:")
    print(f"   - Modèle: {output_dir}/model_classifier.pkl")
    print(f"   - Métriques: {output_dir}/metrics.txt")
    print(f"   - Confusion matrix: confusion_matrix.png")
    
    return model, metrics, X_test, y_test, y_pred


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    # Exécuter le pipeline complet
    model, metrics, X_test, y_test, y_pred = train_classification_pipeline(
        csv_path='tickets_cleaned.csv',
        embeddings_path='data/embeddings/tickets_embeddings.npy',
        model_type='logistic_regression',  # ou 'random_forest'
        test_size=0.2,
        output_dir='models'
    )
    
    # Test de prédiction sur un nouveau ticket (optionnel)
    print("\n" + "="*70)
    print(" TEST DE PRÉDICTION ".center(70))
    print("="*70)
    
    from sentence_transformers import SentenceTransformer
    
    # Charger le modèle d'encodage
    encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Nouveau ticket de test
    new_ticket = "my email is not working, cannot login to outlook"
    
    prediction, probabilities = predict_new_ticket(model, encoder, new_ticket)
    
    print(f"\n🔮 Ticket: '{new_ticket}'")
    print(f"\n✅ Type prédit: {prediction}")
    print(f"\n📊 Probabilités:")
    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cls}: {prob:.4f} ({prob*100:.2f}%)")