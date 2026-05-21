# 🍛 Indian Food Recipe Generator & Image Search

An AI-powered recipe generator that attempts to identify Indian regional foods from uploaded images and fetch their authentic recipes. 

This project uses **OpenAI's CLIP model** to extract image embeddings, **FAISS** for fast similarity search, and a **Streamlit** frontend to provide an interactive user experience.

---

## ⚠️ The Core Challenge & Call for Contributions

While the system is fully functional and successfully fetches recipes, **it struggles with accurately identifying specific regional Indian dishes**. For example, uploading a picture of *Chole Bhature* might incorrectly yield a recipe for a *Gujarati Methi* dish. The recipe provided is correct for the detected dish, but the initial visual detection is inaccurate.

### Why is this happening?
1. **Lack of Diverse Datasets**: Popular open-source datasets like `Food-101` almost entirely lack regional Indian cuisines. 
2. **Proprietary Data**: While companies like Swiggy have recently built custom datasets with over 62,000 Indian food images, these are proprietary and not publicly available.
3. **Fine-tuning Limitations**: Fine-tuning traditional CNNs (like EfficientNet) or Vision Transformers requires massive amounts of high-quality, labeled data, which currently does not exist in the open-source domain for diverse Indian foods.

**Got an idea? I need your help!**
If you know of a better approach, a hidden dataset, or a robust way to solve this specific Indian food classification problem, please reach out to me! 
👉 **[Connect with me on LinkedIn to discuss solutions](https://www.linkedin.com/in/YOUR-LINKEDIN-PROFILE)**

---

## 🛠️ Tech Stack
* **Python**: Core programming language
* **Hugging Face `transformers`**: Used to load `clip-vit-base-patch32` for zero-shot image and text embeddings.
* **PyTorch / Torchvision**: Backend inference engine for the CLIP model.
* **FAISS (Facebook AI Similarity Search)**: Used to build an `IndexFlatIP` vector database for lightning-fast cosine similarity lookups.
* **Pandas & NumPy**: For data manipulation and matrix operations.
* **Streamlit**: For building the interactive web application UI.

## ⚙️ How It Works (The Workflow)

1. **Dataset Preparation**: 
   - A dataset consisting of Indian regional food names, their step-by-step cooking instructions, and corresponding images is parsed via a CSV file.
2. **Embedding Extraction (`build_index.py`)**: 
   - Each image in the dataset is processed through the **CLIP** vision model.
   - The model generates a 512-dimensional vector representation of the image.
   - These vectors are L2-normalized so that inner-product operations equate to cosine similarity.
3. **Vector Database Creation**: 
   - The normalized embeddings are loaded into a **FAISS** index (`recipe_index.faiss`) for highly optimized similarity search.
   - The valid metadata (recipes and names) is saved as a clean CSV file.
4. **User Interaction (`app.py` & `query.py`)**: 
   - A user uploads an image of a dish via the Streamlit web app.
   - The app runs the uploaded image through the CLIP model to generate its vector.
   - FAISS searches the index to find the closest matching vector from our dataset.
   - The app retrieves the corresponding dish name and recipe from the CSV and displays it to the user.

## 🚀 Getting Started

### 0. Download Dataset
Download the [5000 Indian Cuisines Dataset with Images](https://www.kaggle.com/datasets/campusx/5000-indian-cuisines-datasetwith-images) and extract it into the `dataset/` directory.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the FAISS Index
Extract embeddings from your dataset and build the vector database. (Note: The first run will download the ~600MB CLIP model from Hugging Face).
```bash
python build_index.py
```

### 3. Run the App
Launch the Streamlit web interface.
```bash
streamlit run app.py
```

---
**Dataset Credits**: [5000 Indian Cuisines Dataset with Images (Kaggle)](https://www.kaggle.com/datasets/campusx/5000-indian-cuisines-datasetwith-images)
