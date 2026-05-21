# query.py  (imported by app.py)
import faiss
import numpy as np
import pandas as pd
from model import get_image_embedding, get_text_embedding

# ── Load once at module import (not on every query) ───────────────────
index   = faiss.read_index('artifacts/recipe_index.faiss')
df      = pd.read_csv('artifacts/recipes_clean.csv')

def find_recipe(image_path, top_k=3):
    """
    Given a food image path, return top_k most similar dishes with recipes.
    Each result: { dish_name, recipe, similarity_score (0–1) }
    """
    query_emb = get_image_embedding(image_path)           # (512,)
    query_emb = query_emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_emb)                         # ensure unit length
    
    scores, indices = index.search(query_emb, top_k)
    # scores[0]: array of cosine similarities, highest first
    # indices[0]: corresponding row positions in df

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 for unfilled slots
        row = df.iloc[idx]
        results.append({
            'dish_name'  : row['name'],    # adjust column name
            'recipe'     : row['instructions'],        # adjust column name
            'similarity' : round(float(score), 3)
        })
    return results

def find_recipe_by_text(text_query, top_k=3):
    """
    Find recipes matching a text description.
    e.g. find_recipe_by_text("spicy chicken curry with coconut milk")
    """
    text_emb = get_text_embedding(text_query)
    text_emb = text_emb.reshape(1, -1).astype(np.float32)
    
    faiss.normalize_L2(text_emb)
    scores, indices = index.search(text_emb, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        results.append({
            'dish_name' : row['name'],
            'recipe'    : row['instructions'],
            'similarity': round(float(score), 3)
        })
    return results