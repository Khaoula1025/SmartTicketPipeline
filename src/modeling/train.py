import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def prepare_features_and_target(df, embeddings, target_column='type', use_metadata=True):
    """
    Préparer les features (X) et la cible (y)
    
    AMÉLIORATION: Combine embeddings + métadonnées
    
    Args:
        df (pd.DataFrame): Dataset
        embeddings (np.ndarray): Matrice d'embeddings
        target_column (str): Nom de la colonne cible
        use_metadata (bool): Inclure ou non les métadonnées
    
    Returns:
        tuple: (X, y, classes, feature_info)
    """
    print(f"\n🔧 Préparation des features et target...")
    
    # ============================================================
    # PARTIE 1: EMBEDDINGS (features textuelles automatiques)
    # ============================================================
    X_embeddings = embeddings
    print(f" Embeddings shape: {X_embeddings.shape}")
    
    # ============================================================
    # PARTIE 2: MÉTADONNÉES (features structurées)
    # ============================================================
    if use_metadata:
        print(f"\n  Ajout des métadonnées...")
        
        # --- Features catégorielles ---
        categorical_features = ['language', 'queue', 'priority']
        
        # Initialiser les encoders
        label_encoders = {}
        X_categorical = []
        
        for col in categorical_features:
            if col in df.columns:
                # Encoder les catégories en nombres
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].fillna('unknown'))
                X_categorical.append(encoded.reshape(-1, 1))
                label_encoders[col] = le
                
                print(f"   • {col}: {len(le.classes_)} catégories")
            else:
                print(f"  Colonne '{col}' manquante")
        
        # Concatener les features catégorielles
        if X_categorical:
            X_categorical = np.hstack(X_categorical)
            print(f"    Features catégorielles: {X_categorical.shape}")
        else:
            X_categorical = np.array([]).reshape(len(df), 0)
        
        # --- Features numériques (optionnel) ---
        # Exemple: longueur du texte, nombre de mots, etc.
        X_numerical = []
        
        if 'text' in df.columns:
            # Longueur du texte
            text_length = df['text'].str.len().fillna(0).values.reshape(-1, 1)
            X_numerical.append(text_length)
            
            # Nombre de mots
            word_count = df['text'].str.split().str.len().fillna(0).values.reshape(-1, 1)
            X_numerical.append(word_count)
            
            print(f"   • text_length: OK")
            print(f"   • word_count: OK")
        
        if X_numerical:
            X_numerical = np.hstack(X_numerical)
            
            # Normaliser les features numériques
            scaler = StandardScaler()
            X_numerical = scaler.fit_transform(X_numerical)
            
            print(f"    Features numériques: {X_numerical.shape}")
        else:
            X_numerical = np.array([]).reshape(len(df), 0)
        
        # ============================================================
        # PARTIE 3: COMBINER TOUTES LES FEATURES
        # ============================================================
        print(f"\n🔗 Combinaison des features...")
        
        # Concatener: [embeddings | catégorielles | numériques]
        X = np.hstack([X_embeddings, X_categorical, X_numerical])
        
        print(f"   📊 Shape finale: {X.shape}")
        print(f"      • Embeddings: {X_embeddings.shape[1]} dimensions")
        print(f"      • Catégorielles: {X_categorical.shape[1]} features")
        print(f"      • Numériques: {X_numerical.shape[1]} features")
        
        # Stocker les informations sur les features
        feature_info = {
            'embedding_dim': X_embeddings.shape[1],
            'categorical_features': categorical_features,
            'label_encoders': label_encoders,
            'total_features': X.shape[1]
        }
    else:
        # Mode simple: embeddings seulement
        X = X_embeddings
        feature_info = {
            'embedding_dim': X_embeddings.shape[1],
            'total_features': X.shape[1]
        }
    
    # ============================================================
    # PARTIE 4: TARGET
    # ============================================================
    y = df[target_column].values
    classes = np.unique(y)
    
    print(f"\n Target: {target_column}")
    print(f"   • Nombre de classes: {len(classes)}")
    print(f"   • Classes: {classes}")
    
    # Distribution des classes
    print(f"\n Distribution des classes:")
    class_counts = pd.Series(y).value_counts()
    for cls, count in class_counts.items():
        percentage = (count / len(y)) * 100
        print(f"   • {cls}: {count} ({percentage:.1f}%)")
    
    return X, y, classes, feature_info


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
    
    print(f" Entraînement terminé")
    
    return model


# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

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