from sentence_transformers import SentenceTransformer

def load_model():
    """
    Charger le modèle Sentence Transformer depuis Hugging Face
    
    Args:
        model_name (str): Nom du modèle Hugging Face
    
    Returns:
        SentenceTransformer: Modèle pré-entraîné
    """
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    print(f"\n🤖 Chargement du modèle: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✅ Modèle chargé")
    print(f"   Dimension des embeddings: {model.get_sentence_embedding_dimension()}")
    
    return model