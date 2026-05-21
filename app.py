# app.py
import streamlit as st
import tempfile, os
from query import find_recipe

st.set_page_config(
    page_title="Indian Food Recipe Finder",
    page_icon="🍛",
    layout="centered"
)

st.title("🍛 Indian Food Recipe Finder")
st.write("Upload a photo of any Indian dish to get the closest matching recipe.")

uploaded = st.file_uploader(
    "Upload a food image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded:
    # Save upload to temp file so we can pass a file path to the model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(tmp_path, caption="Your image", use_column_width=True)

    with st.spinner("Finding closest dish using CLIP + FAISS..."):
        results = find_recipe(tmp_path, top_k=3)

    best = results[0]

    with col2:
        st.subheader(best['dish_name'])
        st.metric("Similarity score", f"{best['similarity']:.2f}")
        if best['similarity'] > 0.85:
            st.success("Very high confidence match")
        elif best['similarity'] > 0.70:
            st.info("Good match")
        else:
            st.warning("Low confidence — image may be unclear")

    st.divider()
    st.subheader("Recipe")
    st.write(best['recipe'])

    if len(results) > 1:
        with st.expander("Other possible matches"):
            for r in results[1:]:
                st.markdown(f"**{r['dish_name']}** — similarity: `{r['similarity']}`")
                st.write(r['recipe'])

    os.unlink(tmp_path)   # clean up temp file