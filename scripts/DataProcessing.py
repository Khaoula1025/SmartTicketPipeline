import pandas as pd
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# importation des ressources NLTK   
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
   data=pd.read_csv(dataset_path)
   data['subject'] = data['subject'].fillna('')
   data['body'] = data['body'].fillna('')
   data['answer'] = data['answer'].fillna('')

   # Créer le texte combiné pour NLP
   data['text'] = data['subject'] + ' ' + data['body']

   # Gérer les tags utiles (< 8% nulls)
   for col in ['tag_2', 'tag_3', 'tag_4']:
          data[col] = data[col].fillna('')

   # SUPPRIMER les colonnes trop vides (> 30% nulls)
   data = data.drop(columns=['tag_5', 'tag_6', 'tag_7', 'tag_8'])
   data=data.drop(columns=['subject', 'body'])
   data['text'] = data['text'].apply(clean_text)
   data['tokens']=data['text'].apply(tokenize_multilingual)
   return data


