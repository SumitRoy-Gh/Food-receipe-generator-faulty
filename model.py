# model.py
import os
import numpy as np
import torch
from PIL import Image

# Redirect Hugging Face cache to D drive (avoids filling up C drive)
os.environ["HF_HOME"] = r"D:\Food receipe\.hf_cache"

# Hugging Face transformers CLIP (PyTorch backend)
from transformers import CLIPModel, CLIPProcessor

# Load CLIP model (downloads ~600MB once, cached locally after)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()  # set to evaluation mode

def get_image_embedding(image_path):
    """
    Returns L2-normalized 512-d CLIP embedding for one image.
    Shape: (512,)
    """
    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        image_embeds = clip_model.visual_projection(vision_outputs.pooler_output)
    embedding = image_embeds.numpy().astype(np.float32)
    # L2 normalize so inner product = cosine similarity in FAISS
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / norm)[0]  # return shape (512,)

def get_text_embedding(text):
    """
    Returns L2-normalized 512-d CLIP embedding for text.
    Shape: (512,)
    """
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**inputs)
        text_embeds = clip_model.text_projection(text_outputs.pooler_output)
    embedding = text_embeds.numpy().astype(np.float32)
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / norm)[0]