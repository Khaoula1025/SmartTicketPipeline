import pandas as pd
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Setup NLTK ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print(" Ressources NLTK OK\n")
except Exception as e:
    print(f" Erreur téléchargement NLTK: {e}")


# --- Functions ---
def clean_text(text):
    """Nettoyage NLP de base"""
    if pd.isna(text) or text == '':
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Suppression ponctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 3. Suppression espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_multilingual(text):
    """Tokenisation avec stopwords anglais ET allemand"""
    if not text or text == '':
        return []
    
    try:
        # Tokenisation
        tokens = word_tokenize(text)
        
        # Combiner stopwords des 2 langues
        stop_en = set(stopwords.words('english'))
        stop_de = set(stopwords.words('german'))
        all_stopwords = stop_en.union(stop_de)
        
        # Filtrer
        tokens = [
            word for word in tokens 
            if word not in all_stopwords and len(word) > 2
        ]
        
        return tokens
    
    except Exception as e:
        print(f" Erreur: {e}")
        return text.split()
    
def dataProcessing(dataset_path):
    """Charge, nettoie et traite les données"""
    print(f"Traitement du fichier: {dataset_path}")
    data = pd.read_csv(dataset_path)
    
    # Remplir les valeurs manquantes
    for col in ['subject', 'body', 'answer']:
        data[col] = data[col].fillna('')

    # Créer le texte combiné pour NLP
    data['text'] = data['subject'] + ' ' + data['body']

    # Gérer les tags utiles
    for col in ['tag_2', 'tag_3', 'tag_4']:
          data[col] = data[col].fillna('')

    # SUPPRIMER les colonnes trop vides
    columns_to_drop = ['tag_5', 'tag_6', 'tag_7', 'tag_8', 'subject', 'body']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Nettoyage et Tokenisation
    data['text'] = data['text'].apply(clean_text)
    data['tokens'] = data['text'].apply(tokenize_multilingual)
    
    # Sauvegarde
    output_path = 'data/processed/tickets_cleaned.csv'
    data.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé: {output_path}")

# --- Main Execution ---
def main():
    # chemin vers votre fichier
    input_dataset = 'data/raw/dataset.csv' 
    dataProcessing(input_dataset)

if __name__ == "__main__":
    main()