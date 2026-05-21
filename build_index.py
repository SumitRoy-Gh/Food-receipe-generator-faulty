# build_index.py
import os
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from model import get_image_embedding

# ── 1. Load dataset ──────────────────────────────────────────────────
df = pd.read_csv(r'dataset\cuisine_updated.csv')

# Adjust these column names to match your actual CSV
IMAGE_URL_COL = 'image_url'  # column with image url
DISH_NAME_COL = 'name'
RECIPE_COL    = 'instructions'

IMAGE_DIR = r'dataset\data'
os.makedirs('artifacts', exist_ok=True)

# ── 2. Extract embeddings ─────────────────────────────────────────────
embeddings   = []
valid_indices = []

# Create a mapping of basename to actual filename in the directory
image_files = os.listdir(IMAGE_DIR)
basename_to_filename = {}
for f in image_files:
    # f looks like '359.Sorakkai_Poriyal_...jpg'
    # we want to map 'Sorakkai_Poriyal_...jpg' to f
    if '.' in f:
        # Split by the first dot to remove the prefix number
        parts = f.split('.', 1)
        if len(parts) == 2:
            basename_to_filename[parts[1]] = f

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
    image_basename = str(row[IMAGE_URL_COL]).split('/')[-1]
    
    # Look up the actual filename from our mapping
    actual_filename = basename_to_filename.get(image_basename)
    
    if not actual_filename:
        continue  # skip rows with missing images
        
    img_path = os.path.join(IMAGE_DIR, actual_filename)
    
    try:
        emb = get_image_embedding(img_path)   # shape (512,)
        embeddings.append(emb)
        valid_indices.append(idx)
    except Exception as e:
        print(f"Skipped {img_path}: {e}")

# ── 3. Stack into matrix ──────────────────────────────────────────────
embedding_matrix = np.array(embeddings, dtype=np.float32)
print(f"Embedding matrix shape: {embedding_matrix.shape}")  # (N, 512)

# ── 4. Save embedding matrix ──────────────────────────────────────────
np.save('artifacts/embeddings.npy', embedding_matrix)

# ── 5. Save clean recipe CSV (only rows with valid images) ────────────
df_clean = df.iloc[valid_indices].reset_index(drop=True)
df_clean.to_csv('artifacts/recipes_clean.csv', index=False)
print(f"Saved {len(df_clean)} valid recipes")

# ── 6. Build FAISS index ──────────────────────────────────────────────
dimension = embedding_matrix.shape[1]  # 512

# IndexFlatIP = exact brute-force search using Inner Product
# Since vectors are already L2-normalized, IP = cosine similarity
# Higher score = more similar (opposite of distance — so results[0] = best match)
index = faiss.IndexFlatIP(dimension)

# Vectors are already normalized in get_image_embedding()
# but normalize again here as a safety guarantee
faiss.normalize_L2(embedding_matrix)
index.add(embedding_matrix)

faiss.write_index(index, 'artifacts/recipe_index.faiss')
print(f"FAISS index built with {index.ntotal} vectors")
print("Done. Run: streamlit run app.py")