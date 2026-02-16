import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Store encoders in a dictionary for easy saving
    feature_info = {
        'encoders': {
            'meta': encoder,
            'tags': mlb,
            'target': y_encoder
        }
    }
    print(f"  • Target '{'type'}' ready. Found {len(classes)} classes.")
    return X, y, classes , feature_info


def train_and_evaluate_model(csv_path, embeddings_path, model_save_path="models/ticket_model.joblib"):
    """
    Splits data, trains a Random Forest, and evaluates performance.
    """
    print(f"\n Starting Model Training Pipeline...")
    df, embeddings=load_data_and_embeddings(csv_path, embeddings_path)
    X,y,classes,feature_info=prepare_features_and_target(df, embeddings)
    # 1. SPLIT DATA
    # Stratify ensures the distribution of classes is similar in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  • Training set: {X_train.shape[0]} samples")
    print(f"  • Testing set: {X_test.shape[0]} samples")

    # 2. TRAIN MODEL
    # Random Forest handles high-dimensional vectors (embeddings) well
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        random_state=42, 
        n_jobs=-1 # Use all CPU cores
    )
    print(f"  • Training Random Forest...")
    model.fit(X_train, y_train)
    print(f"  • Model trained successfully.")
    # SAVE EVERYTHING
    payload = {
        'model': model,
        'feature_info': feature_info
    }
    joblib.dump(payload, model_save_path)
    print(f"✅ Model and encoders saved to {model_save_path}")
    # 3. PREDICT & EVALUATE
    y_pred = model.predict(X_test)
    
    print(f"\n📊 Evaluation Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # 4. PLOT CONFUSION MATRIX
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    return model
# model =train_and_evaluate_model(csv_path='data/processed/tickets_cleaned.csv', embeddings_path='data/embeddings/tickets_embeddings.npy', model_save_path="models/ticket_model.joblib")
# print(f"\n✅ Training pipeline completed successfully.")
# print(model)
# 1. Define your paths
# csv_path = 'data/processed/tickets_cleaned.csv'
# embeddings_path = 'data/embeddings/tickets_embeddings.npy'

# # 2. Use your existing loader function to get the actual objects
# df_loaded, embeddings_loaded = load_data_and_embeddings(csv_path, embeddings_path)

# # 3. Pass the LOADED objects into the feature preparation function
# # Note: I've swapped them to match your function's argument order (df, embeddings)
# X, y, classes, feature_info = prepare_features_and_target(df_loaded, embeddings_loaded)

# # 4. Now you can safely inspect feature_info
# print("\n--- Feature Info Encoders ---")
# print(feature_info)