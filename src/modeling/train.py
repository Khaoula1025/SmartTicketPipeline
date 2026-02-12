import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
    
    print(f"Cohérence vérifiée")
    
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
    print(f"\n Préparation des features et target...")
    
    # Features = embeddings
    X = embeddings
    
    # Target = type de ticket
    y = df[target_column].values
    
    # Classes uniques
    classes = np.unique(y)
    
    print(f" Features shape: {X.shape}")
    print(f" Target shape: {y.shape}")
    print(f" Nombre de classes: {len(classes)}")
    print(f"   Classes: {classes}")
    
    # Distribution des classes
    print(f"\n Distribution des classes:")
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
