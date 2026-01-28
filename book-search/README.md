# Book Search (Gutenberg)

Download Project Gutenberg books as plain text and index them into Elasticsearch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download books

```bash
python download_books.py --count 100 --out-dir data
```

This creates:
- `data/books/` with `.txt` files
- `data/metadata.jsonl` with `id`, `title`, `authors`, `year`, `text_path`

## Build Whoosh index

```bash
python build_index.py --index-dir data/index --data-dir data
```

The index contains one document per book with `title`, `authors`, `year`, and `text` fields.

## Search

```bash
python search_books.py --index-dir data/index --query "whale ship" --limit 5
```

## Streamlit UI

```bash
streamlit run app.py
```
