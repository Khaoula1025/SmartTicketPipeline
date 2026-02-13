import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder


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
    print(f"Dataset: {len(df)} lignes")
    
    # Charger les embeddings
    embeddings = np.load(embeddings_path)
    print(f" Embeddings: {embeddings.shape}")
    
    # Vérifier la cohérence
    if len(df) != len(embeddings):
        raise ValueError(
            f"Erreur: tailles différentes! "
            f"Dataset: {len(df)}, Embeddings: {len(embeddings)}"
        )
    
    print(f" Cohérence vérifiée")
    
    return df, embeddings


def prepare_features_and_target(df, embeddings):
    """
    Minimalist feature engineering: Combines text embeddings, 
    one-hot encoded metadata, and multi-label tags.
    """
    print(f"\n Preparing Features for X (Input) and y (Target)...")

    # 1. PROCESS METADATA (Language, Priority, Queue)
    # Using OneHotEncoder is better for ML models than LabelEncoder for non-ordinal data
    cat_cols = ['language', 'priority', 'queue']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_meta = encoder.fit_transform(df[cat_cols].fillna('unknown'))
    print(f"  • Categorical features encoded: {X_meta.shape[1]} columns")

    # 2. PROCESS TAGS (Multi-label)
    # Combine tag_1 through tag_4 into a list of tags per row
    tag_cols = ['tag_1', 'tag_2', 'tag_3', 'tag_4']
    combined_tags = df[tag_cols].fillna('').apply(
        lambda x: [t.strip() for t in x if t.strip() != ''], axis=1
    )
    
    mlb = MultiLabelBinarizer()
    X_tags = mlb.fit_transform(combined_tags)
    print(f"  • Tags encoded (Multi-label): {X_tags.shape[1]} columns")

    # 3. COMBINE EVERYTHING INTO X
    # Structure: [Embeddings | OneHot Metadata | Multi-label Tags]
    X = np.hstack([embeddings, X_meta, X_tags])
    print(f"  • Final X shape: {X.shape}")

    # 4. PREPARE TARGET (y)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df['type'])
    classes = y_encoder.classes_
    
    print(f"  • Target '{'type'}' ready. Found {len(classes)} classes.")
    
    # feature_info = {
    #     'total_dim': X.shape[1],
    #     'embedding_dim': embeddings.shape[1],
    #     'meta_dim': X_meta.shape[1],
    #     'tags_dim': X_tags.shape[1],
    #     'encoders': {'meta': encoder, 'tags': mlb, 'target': y_encoder}
    # }

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
    print(f"\n Séparation train/test...")
    print(f"   Test size: {test_size*100:.0f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Garde la même distribution de classes
    )
    
    print(f" Split terminé:")
    print(f"   • Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   • Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
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
    print(f"\n Entraînement du modèle: {model_type}...")
    
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
    
    print(f" Entraînement terminé")
    
    return model

def example_usage():
    """Exemple montrant la différence entre les deux approches"""
    
    print("="*70)
    print(" COMPARAISON: Embeddings seuls vs Embeddings + Métadonnées ".center(70))
    print("="*70)
    
    # Charger les données
    df, embeddings = load_data_and_embeddings(
        'data/processed/tickets_cleaned.csv',
        'data/embeddings/tickets_embeddings.npy'
    )
    
    # ============================================================
    # APPROCHE 1: Embeddings seulement (code original)
    # ============================================================
    print("\n" + "="*70)
    print(" APPROCHE 1: Embeddings seulement ".center(70))
    print("="*70)
    
    X1, y1, classes1, info1 = prepare_features_and_target(
        df, embeddings, use_metadata=False
    )
    print(f"\n📊 Résultat:")
    print(f"   X shape: {X1.shape}")
    print(f"   Features: {info1['total_features']}")
    
    # ============================================================
    # APPROCHE 2: Embeddings + Métadonnées (amélioré)
    # ============================================================
    print("\n" + "="*70)
    print(" APPROCHE 2: Embeddings + Métadonnées ".center(70))
    print("="*70)
    
    X2, y2, classes2, info2 = prepare_features_and_target(
        df, embeddings, use_metadata=True
    )
    print(f"\n📊 Résultat:")
    print(f"   X shape: {X2.shape}")
    print(f"   Features: {info2['total_features']}")
    
    # ============================================================
    # COMPARAISON
    # ============================================================
    print("\n" + "="*70)
    print(" DIFFÉRENCE ".center(70))
    print("="*70)
    print(f"Approche 1: {X1.shape[1]} features (embeddings seulement)")
    print(f"Approche 2: {X2.shape[1]} features (embeddings + métadonnées)")
    print(f"Gain: +{X2.shape[1] - X1.shape[1]} features supplémentaires")
    

if __name__ == "__main__":
    example_usage()