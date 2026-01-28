# Book Search (Gutenberg)

Download Project Gutenberg books as plain text and index them into Whoosh.

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

## How it works

### 1) Data download
- `download_books.py` fetches Project Gutenberg texts and writes them to `data/books/`.
- It also writes `data/metadata.jsonl`, where each line is a JSON object containing:
  - `id`, `title`, `authors`, `year`, `text_path`

### 2) Indexing pipeline
- `build_index.py` creates a Whoosh index at `data/index/` if it doesn't exist.
- For each record in `metadata.jsonl`, it:
  - loads the raw text file
  - strips the Gutenberg boilerplate header/footer when present
  - normalizes text to improve tokenization:
    - fixes hyphenated line breaks (e.g., `word-\nwrap` → `wordwrap`)
    - collapses soft line wraps into spaces
    - normalizes whitespace
  - stores one document per book with `id`, `title`, `authors`, `year`, `text`
- The `text` field uses a stemming analyzer pipeline:
  - `RegexTokenizer()` splits text into word-like tokens based on a regex pattern.
  - `LowercaseFilter()` lowercases every token so case does not affect matching.
  - `StopFilter()` removes common stopwords (e.g., “the”, “and”) to reduce noise.
  - `StemFilter()` reduces tokens to their stem so inflections match (e.g., “sailing” → “sail”).
  - Together this helps queries like “sailing” match “sailor/sailed”.

#### Why normalize text?
Project Gutenberg texts are formatted for reading, not search. Normalization makes tokenization and scoring more consistent:
- **Hyphenated line breaks**: Gutenberg often splits words at line ends (`word-\nwrap`). If left intact, the tokenizer would treat this as two tokens. We merge them back to the original word to preserve matching.
- **Soft line wraps**: Plain text line wrapping inserts newlines mid‑sentence. Collapsing single newlines into spaces prevents accidental token boundaries and improves phrase matching.
- **Whitespace normalization**: Collapsing repeated spaces and excessive blank lines reduces index noise and improves snippet quality.

### 3) Searching (CLI + UI)
- `search_books.py` and `app.py` both:
  - parse queries across `title`, `authors`, `text`
  - apply field boosts so title/authors dominate matches
  - use BM25F for scoring
- Default boosts:
  - `title`: 3.0
  - `authors`: 2.0
  - `text`: 1.0

### 4) Rebuilding the index
- If you change analyzers or normalization logic in `build_index.py`, rebuild the index:

```bash
python build_index.py --index-dir data/index --data-dir data
```
