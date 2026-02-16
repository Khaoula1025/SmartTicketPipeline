import numpy as np
import pandas as pd
import joblib
from src.embeddings.hf_model import load_model

def predict_new_tickets(new_df, model, feature_info, embedding_model):
    print(f"\n Predicting new tickets...")
    
    # 1. Generate Embeddings (Using the ACTUAL model instance)
    new_embeddings = embedding_model.encode(
        new_df['text'].tolist(), 
        show_progress_bar=False, 
        convert_to_numpy=True
    )
    
    # 2. Process Metadata
    X_meta = feature_info['encoders']['meta'].transform(
        new_df[['language', 'priority', 'queue']].fillna('unknown')
    )
    
    # 3. Process Tags
    tag_cols = ['tag_1', 'tag_2', 'tag_3', 'tag_4']
    combined_tags = new_df[tag_cols].fillna('').apply(
        lambda x: [t.strip() for t in x if t.strip() != ''], axis=1
    )
    X_tags = feature_info['encoders']['tags'].transform(combined_tags)
    
    # 4. Concatenate & Predict
    X_new = np.hstack([new_embeddings, X_meta, X_tags])
    predictions = model.predict(X_new)
    
    # 5. Inverse Transform
    return feature_info['encoders']['target'].inverse_transform(predictions)

# --- CORRECTED EXAMPLE USAGE ---
if __name__ == "__main__":
    # 1. Load the saved assets
    saved_data = joblib.load("models/ticket_model.joblib")
    
    # 2. Call the function to get the actual SentenceTransformer instance
    # FIX: Notice the () at the end of load_model()
    bert_model = load_model() 

    new_ticket = pd.DataFrame([{
        'text': 'The application crashes when I try to save',
        'language': 'en',
        'priority': 'High',
        'queue': 'Development',
        'tag_1': 'Crash', 'tag_2': 'Bug', 'tag_3': '', 'tag_4': ''
    }])

    result = predict_new_tickets(
        new_df=new_ticket, 
        model=saved_data['model'], 
        feature_info=saved_data['feature_info'], 
        embedding_model=bert_model # Now passing the object, not the function
    )
    
    print(f"Predicted Class: {result[0]}")