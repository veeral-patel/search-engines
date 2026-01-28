import tempfile
from pathlib import Path

import duckdb
import numpy as np
import streamlit as st
from PIL import Image

from clip_utils import encode_image, encode_text, load_clip


@st.cache_resource
def get_clip_bundle(model_name: str, pretrained: str):
    return load_clip(model_name=model_name, pretrained=pretrained)


@st.cache_data
def load_db_embeddings(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("SELECT id, path, embedding FROM images ORDER BY id").fetchall()
    meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
    con.close()
    if not rows:
        return [], [], np.zeros((0, 1), dtype=np.float32), meta
    ids, paths, embeds = zip(*rows)
    matrix = np.array(embeds, dtype=np.float32)
    return list(ids), list(paths), matrix, meta


def render_results(paths, scores, top_k):
    top_k = min(top_k, len(scores))
    top_idx = np.argsort(-scores)[:top_k]

    cols = st.columns(3)
    for rank, i in enumerate(top_idx, start=1):
        col = cols[(rank - 1) % 3]
        with col:
            st.image(paths[i], use_container_width=True)
            st.caption(f"#{rank:02d} score={scores[i]:.4f}")


def main():
    st.set_page_config(page_title="Image Search", layout="wide")
    st.title("Image Search with OpenCLIP + DuckDB")

    with st.sidebar:
        st.header("Settings")
        db_path = st.text_input("DuckDB path", value="data/embeddings.duckdb")
        model_name = st.text_input("CLIP model", value="ViT-B-32")
        pretrained = st.text_input("Pretrained", value="openai")
        top_k = st.slider("Top K", min_value=1, max_value=20, value=6)

    db_file = Path(db_path)
    if not db_file.exists():
        st.error("DuckDB not found. Run embed_images.py first.")
        st.stop()

    ids, paths, matrix, meta = load_db_embeddings(db_path)
    if not paths:
        st.error("No embeddings in DB. Run embed_images.py first.")
        st.stop()

    if meta:
        meta_model = meta.get("model_name")
        meta_pretrained = meta.get("pretrained")
        if meta_model and meta_model != model_name:
            st.warning(
                f"DB model is {meta_model}, but UI is set to {model_name}."
            )
        if meta_pretrained and meta_pretrained != pretrained:
            st.warning(
                f"DB pretrained is {meta_pretrained}, but UI is set to {pretrained}."
            )

    bundle = get_clip_bundle(model_name, pretrained)

    tab_image, tab_text = st.tabs(["Image Query", "Text Query"])

    with tab_image:
        st.subheader("Find similar images")
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp.write(uploaded.getbuffer())
            temp.close()

            query_img = Image.open(temp.name).convert("RGB")
            st.image(query_img, caption="Query image", use_container_width=False)

            query_vec = encode_image(bundle, temp.name)
            scores = matrix @ query_vec
            render_results(paths, scores, top_k)

    with tab_text:
        st.subheader("Search by text")
        text = st.text_input("Describe what you want to find")
        if text:
            query_vec = encode_text(bundle, text)
            scores = matrix @ query_vec
            render_results(paths, scores, top_k)


if __name__ == "__main__":
    main()
