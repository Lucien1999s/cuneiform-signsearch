# Cuneiform Tablet Semantic Search

Text-first semantic search over translated cuneiform tablet entries. CPU-friendly and easy to extend.

## Setup

1) Install dependencies
```
pip install -r requirements.txt
```

2) Build the text index (runs once per model)
```
python3 src/text_index.py --output data --collection tablets --model BAAI/bge-large-en-v1.5
```

## Run the Demo UI

```
python3 main.py
```

## Query from CLI

```
python3 src/text_search.py "trade of barley with merchant" --output data --collection tablets --top-k 5 --model BAAI/bge-large-en-v1.5
```

## Notes

- Dataset images live in `datasets/`.
- Milvus Lite index is stored in `data/milvus.db`.
- Embedding model cache is stored in `models/hf/`.
