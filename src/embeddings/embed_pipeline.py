import pandas as pd
import numpy as np
import os
import chromadb
from tqdm import tqdm
from src.embeddings.hf_model import load_model
def load_cleaned_data(filepath):
    """
    Charger le dataset nettoyé
    
    Args:
        filepath (str): Chemin vers le fichier CSV nettoyé
    
    Returns:
        pd.DataFrame: Dataset nettoyé
    """
    print(f"\n Chargement du dataset: {filepath}")
    df = pd.read_csv(filepath)
    print(f" {len(df)} lignes chargées")
    # Vérification
    if 'text' not in df.columns:
        raise ValueError("Colonne 'text' manquante! Exécutez d'abord l'étape 1.")
    
    return df


def generate_embeddings(df, model=load_model, text_column='text', batch_size=32):
    """
    Générer les embeddings pour tous les textes
    
    Args:
        df (pd.DataFrame): Dataset avec les textes
        model (SentenceTransformer): Modèle pour encoder
        text_column (str): Nom de la colonne contenant le texte
        batch_size (int): Taille du batch pour l'encodage
    
    Returns:
        np.ndarray: Matrice d'embeddings (n_samples, embedding_dim)
    """
    print(f"\n Génération des embeddings...")
    
    # Extraire les textes
    texts = df[text_column].tolist()
    
    # Encoder tous les textes
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Embeddings générés: {embeddings.shape}")
    
    return embeddings


def normalize_embeddings(embeddings):
    """
    Normaliser les vecteurs avec la norme L2
    
    Args:
        embeddings (np.ndarray): Matrice d'embeddings
    
    Returns:
        np.ndarray: Embeddings normalisés
    """
    print(f"\n Normalisation des embeddings...")
    
    # Normalisation L2
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    
    # Vérification
    verification_norms = np.linalg.norm(embeddings_normalized, axis=1)
    print(f"Normalisation terminée")
    print(f"   Norme moyenne: {np.mean(verification_norms):.6f}")
    print(f"   Norme min: {np.min(verification_norms):.6f}")
    print(f"   Norme max: {np.max(verification_norms):.6f}")
    
    return embeddings_normalized


def save_embeddings(embeddings, df, output_dir='data/embeddings'):
    """
    Sauvegarder les embeddings et métadonnées
    
    Args:
        embeddings (np.ndarray): Matrice d'embeddings
        df (pd.DataFrame): Dataset avec métadonnées
        output_dir (str): Dossier de sortie
    
    Returns:
        tuple: (embeddings_path, metadata_path)
    """
    print(f"\n Sauvegarde des embeddings...")
    
    # Créer le dossier si nécessaire5
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les embeddings
    embeddings_path = os.path.join(output_dir, 'tickets_embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f" Embeddings: {embeddings_path}")
    print(f"   Taille: {os.path.getsize(embeddings_path) / 1024 / 1024:.2f} MB")
    
    # Sauvegarder les métadonnées
    metadata = df[['type', 'queue', 'priority', 'language', 'tag_1', 'tag_2', 'tag_3', 'tag_4']].copy()
    metadata['text_preview'] = df['text'].str[:100]
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata.to_csv(metadata_path, index=False)
    print(f" Métadonnées: {metadata_path}")
    
    return embeddings_path, metadata_path


def index_in_chromadb(embeddings, df, chroma_path='data/chromadb', collection_name='ticket_embeddings'):
    """
    Indexer les embeddings dans ChromaDB
    
    Args:
        embeddings (np.ndarray): Matrice d'embeddings normalisés
        df (pd.DataFrame): Dataset avec métadonnées
        chroma_path (str): Chemin vers la base ChromaDB
        collection_name (str): Nom de la collection
    
    Returns:
        chromadb.Collection: Collection ChromaDB
    """
    print(f"\n  Indexation dans ChromaDB...")
    
    # Initialiser le client ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Supprimer l'ancienne collection si elle existe
    try:
        client.delete_collection(name=collection_name)
        print(f" Ancienne collection '{collection_name}' supprimée")
    except:
        pass
    
    # Créer la nouvelle collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "IT Support Ticket Embeddings"}
    )
    
    # Préparer les données
    ids = [f"ticket_{i}" for i in range(len(df))]
    documents = df['text'].tolist()
    metadatas = df[['type', 'queue', 'priority', 'language','tag_1', 'tag_2', 'tag_3', 'tag_4']].to_dict('records')
    
    # Indexer par batches (limite ChromaDB)
    batch_size = 5000
    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    
    print(f"   Indexation en {num_batches} batch(es)...")
    
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Indexation"):
        batch_end = min(i + batch_size, len(embeddings))
        
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    # Vérification
    count = collection.count()
    print(f" {count} documents indexés dans ChromaDB")
    print(f"   Chemin: {chroma_path}")
    
    return collection


def test_similarity_search(collection, query_text, model, top_k=5):
    """
    Tester la recherche de similarité
    
    Args:
        collection: Collection ChromaDB
        query_text (str): Texte de recherche
        model (SentenceTransformer): Modèle pour encoder la requête
        top_k (int): Nombre de résultats à retourner
    """
    print(f"\n🔍 Test de recherche de similarité...")
    print(f"   Requête: '{query_text}'")
    
    # Encoder la requête
    query_embedding = model.encode([query_text])[0]
    
    # Rechercher les documents similaires
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    # Afficher les résultats
    print(f"\n📋 Top {top_k} tickets similaires:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n   {i+1}. Type: {metadata['type']} | Distance: {distance:.4f}")
        print(f"      Langue: {metadata['language']} | Queue: {metadata['queue']}")
        print(f"      Texte: {doc[:120]}...")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def generate_embeddings_pipeline(
    input_csv='data/processed/tickets_cleaned.csv',
    output_dir='data/embeddings',
    chroma_path='data/chromadb',
    batch_size=32
):
    """
    Pipeline complet de génération d'embeddings
    
    Args:
        input_csv (str): Chemin vers le dataset nettoyé
        model_name (str): Nom du modèle Hugging Face
        output_dir (str): Dossier de sortie pour les embeddings
        chroma_path (str): Chemin vers ChromaDB
        batch_size (int): Taille du batch pour l'encodage
    
    Returns:
        tuple: (embeddings, df, model, collection)
    """
    print("="*70)
    print(" GÉNÉRATION D'EMBEDDINGS ".center(70))
    print("="*70)
    
    # 1. Charger les données
    df = load_cleaned_data(input_csv)
    
    # 2. Charger le modèle
    model = load_model()
    
    # 3. Générer les embeddings
    embeddings = generate_embeddings(df, model, batch_size=batch_size)
    
    # 4. Normaliser les embeddings
    embeddings_normalized = normalize_embeddings(embeddings)
    
    # 5. Sauvegarder
    save_embeddings(embeddings_normalized, df, output_dir)
    
    # 6. Indexer dans ChromaDB
    collection = index_in_chromadb(embeddings_normalized, df, chroma_path)
    
    # 7. Test de recherche
    test_similarity_search(collection, "email not working login problem", model)
    
    print("\n" + "="*70)
    print(" ✅ ÉTAPE 2 TERMINÉE ".center(70))
    print("="*70)
    print(f"\n📊 Résumé:")
    print(f"   - Embeddings générés: {len(embeddings_normalized)}")
    print(f"   - Dimension: {embeddings_normalized.shape[1]}")
    print(f"   - Fichier: {output_dir}/tickets_embeddings.npy")
    print(f"   - ChromaDB: {chroma_path}")
    
    return embeddings_normalized, df, model, collection


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    # Exécuter le pipeline complet
    embeddings, df, model, collection = generate_embeddings_pipeline(
        input_csv='data/processed/tickets_cleaned.csv',
        output_dir='data/embeddings',
        chroma_path='data/chromadb',
        batch_size=32
    )