import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Paths and constants
DATA_FILE = "data/messages.json"
EMB_FILE = "data/embeddings.npy"
META_FILE = "data/meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, and accurate


def load_or_download_model(model_name=MODEL_NAME):
    """
    Load the SentenceTransformer model.
    If it's not cached, try to download it automatically.
    If download fails (e.g., no internet), raise a clean error.
    """
    try:
        print(f"Loading model '{model_name}' ...")
        model = SentenceTransformer(model_name)
        print(f"✅ Model '{model_name}' loaded successfully")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load model '{model_name}' normally. Trying to download again...")
        try:
            model = SentenceTransformer(model_name, cache_folder=".model_cache")
            print(f"✅ Model '{model_name}' downloaded successfully")
            return model
        except Exception as e2:
            raise RuntimeError(
                f"❌ Could not load or download model '{model_name}'. "
                f"Please check your internet connection or manually download it using:\n"
                f"    from sentence_transformers import SentenceTransformer\n"
                f"    SentenceTransformer('{model_name}')"
            ) from e2


def build_index(model_name=MODEL_NAME):
    """Builds the embedding index for the messages dataset."""
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError("⚠️ Run fetch_messages.py first to create 'data/messages.json'")

    with open(DATA_FILE, "r", encoding="utf8") as f:
        items = json.load(f)

    texts, meta = [], []
    for it in items:
        text = f"{it.get('user_name', '')} | {it.get('timestamp', '')} | {it.get('message', '')}"
        texts.append(text)
        meta.append({
            "id": it.get("id"),
            "user_id": it.get("user_id"),
            "user_name": it.get("user_name"),
            "timestamp": it.get("timestamp")
        })

    print(f"🧠 Encoding {len(texts)} messages with model '{model_name}' ...")

    # Load or auto-download model
    model = load_or_download_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings and metadata
    os.makedirs("data", exist_ok=True)
    np.save(EMB_FILE, embeddings)
    with open(META_FILE, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved embeddings to {EMB_FILE}")
    print(f"✅ Saved metadata to {META_FILE}")

    return EMB_FILE, META_FILE


if __name__ == "__main__":
    build_index()
