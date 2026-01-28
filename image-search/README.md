# Image Search with OpenCLIP + DuckDB

This repo contains a minimal image search pipeline:
1. Download a batch of images.
2. Embed them with OpenCLIP.
3. Store embeddings in DuckDB.
4. Query by image or text.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: OpenCLIP will download model weights the first time you run `embed_images.py` (network required).

## 1) Download images

```bash
python download_images.py --count 100 --out-dir data/images
```

Options:
- `--count`: number of images (default 100)
- `--size`: square size in pixels (default 512)
- `--seed`: seed for URL generation (default 42)

## 2) Embed images into DuckDB

```bash
python embed_images.py --image-dir data/images --db data/embeddings.duckdb
```

Options:
- `--model`: OpenCLIP model (default `ViT-B-32`)
- `--pretrained`: pretrained weights (default `openai`)
- `--reset`: drop/rebuild tables

## 3) Query similar images by image

```bash
python query_image.py data/images/image_0000.jpg --top-k 5
```

## 4) Query similar images by text

```bash
python query_text.py "a snowy mountain landscape" --top-k 5
```

## Streamlit UI

```bash
streamlit run app.py
```

## Notes

- Embeddings are stored as `FLOAT[]` in DuckDB.
- `embed_images.py` overwrites the `images` table each run.
- Model/pretrained metadata is stored in the `metadata` table to prevent accidental mismatch.
