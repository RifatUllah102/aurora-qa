# # qa.py
# import json, os, numpy as np, re
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List, Dict

# DATA_FILE = "data/messages.json"
# EMB_FILE = "data/embeddings.npy"
# META_FILE = "data/meta.json"

# # Simple utility: load messages
# def load_messages():
#     with open(DATA_FILE, "r", encoding="utf8") as f:
#         return json.load(f)

# def load_meta_and_embeddings():
#     meta = json.load(open(META_FILE, "r", encoding="utf8"))
#     embeddings = np.load(EMB_FILE)
#     return meta, embeddings

# # Keyword-based quick answerer (fallback)
# def keyword_fallback(question: str, messages: List[Dict]) -> str:
#     # q = question.lower()
#     # # look for names (naive): capitalized tokens in question
#     # # e.g., "When is Layla planning her trip to London?"
#     # name_matches = re.findall(r"\b[A-Z][a-z]{1,20}\b", question)
#     # # remove words that are likely not names
#     # stop_names = {"When","What","How","Where","Why","Which","Is","Are","The","A","An"}
#     # candidates = [n for n in name_matches if n not in stop_names]
#     # # Simple heuristics:
#     # if "trip" in q or "travel" in q or "flight" in q:
#     #     person = candidates[0] if candidates else None
#     #     if person:
#     #         # find messages by that user_name or messages mentioning the person + city/date patterns
#     #         for m in messages:
#     #             if m.get("user_name") == person or person.lower() in m.get("message","").lower():
#     #                 msg = m.get("message","")
#     #                 # naive date extractor (YYYY or Month names + day)
#     #                 date = re.search(r"\b(?:Jan(?:uary)?|Feb|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\w\s,.-]{0,15}\d{1,2}|\b\d{4}\b", msg)
#     #                 if date:
#     #                     return f"{person} — {msg}"
#     #         return f"No explicit trip date found for {person} in the cached messages."
#     # # cars / number questions
#     # if "car" in q or "cars" in q:
#     #     person = candidates[0] if candidates else None
#     #     if person:
#     #         # search for patterns like "I have 2 cars", "two cars", "1 car"
#     #         pat = re.compile(r"(\b\d+\b|\bone\b|\btwo\b|\bthree\b|\bfour\b)\s+(car|cars)\b", re.I)
#     #         for m in messages:
#     #             if m.get("user_name") == person or person.lower() in m.get("message","").lower():
#     #                 mo = pat.search(m.get("message",""))
#     #                 if mo:
#     #                     return f"{person} — {mo.group(0)} (message: \"{m.get('message')}\")"
#     #         return f"No explicit 'car' count found for {person} in the cached messages."
#     # # favorites / list queries (restaurants etc.)
#     # if "favorite" in q or "favourite" in q or "favorite restaurants" in q or "restaurants" in q:
#     #     person = candidates[0] if candidates else None
#     #     if person:
#     #         found = []
#     #         for m in messages:
#     #             if m.get("user_name") == person or person.lower() in m.get("message","").lower():
#     #                 # naive restaurant words detection: "restaurant", "dined at", "love", "like"
#     #                 if "restaurant" in m.get("message","").lower() or "dined" in m.get("message","").lower() or "favorite" in m.get("message","").lower():
#     #                     found.append(m.get("message"))
#     #         if found:
#     #             return f"{person} — found mentions:\n" + "\n".join(found[:5])
#     #         return f"No favorite restaurants found for {person} in the cached messages."
#     # return ""  # no fallback answer
#     q = question.lower()
#     # naive extraction of capitalized tokens as possible names
#     name_matches = re.findall(r"\b[A-Z][a-z]{1,20}\b", question)
#     stop_names = {"When","What","How","Where","Why","Which","Is","Are","The","A","An"}
#     candidates = [n for n in name_matches if n not in stop_names]

#     if not candidates:
#         return ""

#     # Combine all user_names in dataset for partial matching
#     all_user_names = [m.get("user_name","") for m in messages]

#     # Try to match question token to full names in dataset
#     person = None
#     for candidate in candidates:
#         # Case-insensitive partial match
#         matches = [name for name in all_user_names if candidate.lower() in name.lower()]
#         if matches:
#             person = matches[0]  # pick the first matching full name
#             break

#     if not person:
#         return ""  # could not find a matching user

#     # Trip / travel questions
#     if "trip" in q or "travel" in q or "flight" in q:
#         for m in messages:
#             if person.lower() in m.get("user_name","").lower() or person.lower() in m.get("message","").lower():
#                 msg = m.get("message","")
#                 # naive date extractor
#                 date = re.search(
#                     r"\b(?:Jan(?:uary)?|Feb|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\w\s,.-]{0,15}\d{1,2}|\b\d{4}\b",
#                     msg)
#                 if date:
#                     return f"{person} — {msg}"
#         return f"No explicit trip date found for {person} in the cached messages."

#     # Cars / number questions
#     if "car" in q or "cars" in q:
#         pat = re.compile(r"(\b\d+\b|\bone\b|\btwo\b|\bthree\b|\bfour\b)\s+(car|cars)\b", re.I)
#         for m in messages:
#             if person.lower() in m.get("user_name","").lower() or person.lower() in m.get("message","").lower():
#                 mo = pat.search(m.get("message",""))
#                 if mo:
#                     return f"{person} — {mo.group(0)} (message: \"{m.get('message')}\")"
#         return f"No explicit 'car' count found for {person} in the cached messages."

#     # Favorites / restaurants
#     if "favorite" in q or "favourite" in q or "restaurants" in q:
#         found = []
#         for m in messages:
#             if person.lower() in m.get("user_name","").lower() or person.lower() in m.get("message","").lower():
#                 if "restaurant" in m.get("message","").lower() or "dined" in m.get("message","").lower() or "favorite" in m.get("message","").lower():
#                     found.append(m.get("message"))
#         if found:
#             return f"{person} — found mentions:\n" + "\n".join(found[:5])
#         return f"No favorite restaurants found for {person} in the cached messages."

#     return ""

# # Semantic search + answer assembly
# def semantic_answer(question: str, top_k=5) -> str:
#     # if embeddings not present, return empty string
#     if not (os.path.exists(EMB_FILE) and os.path.exists(META_FILE)):
#         return ""
#     import numpy as np
#     from sentence_transformers import SentenceTransformer
#     meta = json.load(open(META_FILE, "r", encoding="utf8"))
#     embeddings = np.load(EMB_FILE)
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     q_emb = model.encode([question], convert_to_numpy=True)
#     sims = cosine_similarity(q_emb, embeddings)[0]
#     top_idx = (-sims).argsort()[:top_k]
#     results = []
#     for idx in top_idx:
#         results.append({
#             "score": float(sims[idx]),
#             "user_name": meta[idx].get("user_name"),
#             "timestamp": meta[idx].get("timestamp"),
#             "text": meta[idx].get("timestamp","") + " | " + meta[idx].get("user_name","") + " | " + "MESSAGE SNIPPET"
#         })
#     # load full message text to show
#     messages = json.load(open(DATA_FILE, "r", encoding="utf8"))
#     answer_texts = []
#     for idx in top_idx:
#         m = messages[idx]
#         answer_texts.append(f"{m.get('user_name')} ({m.get('timestamp')}): {m.get('message')}")
#     # assemble concise answer: use the highest scoring message as the "answer"
#     best = messages[top_idx[0]]
#     combined = "Most relevant message:\n" + answer_texts[0] + "\n\nOther relevant messages:\n" + "\n".join(answer_texts[1:3])
#     return combined

# # Top-level: answer(question) -> str
# def answer(question: str) -> str:
#     messages = load_messages()
#     # 1) Try keyword fallback heuristics
#     k = keyword_fallback(question, messages)
#     if k:
#         return k
#     # 2) Try semantic search if available
#     sem = semantic_answer(question)
#     if sem:
#         return sem
#     # 3) Last resort: simple substring search
#     q = question.lower()
#     hits = [m for m in messages if q.split()[0] in (m.get("message","").lower())][:5]
#     if hits:
#         return "Found messages:\n" + "\n".join([f"{h.get('user_name')}: {h.get('message')}" for h in hits])
#     return "Sorry — I couldn't find an answer in the dataset."


# # qa.py
# import json
# import os
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer
# import re

# # Paths to data and embeddings
# DATA_FILE = "data/messages.json"
# EMB_FILE = "data/embeddings.npy"
# META_FILE = "data/meta.json"

# # Load messages from JSON
# def load_messages() -> List[Dict]:
#     with open(DATA_FILE, "r", encoding="utf8") as f:
#         return json.load(f)

# # Load metadata and embeddings
# def load_meta_and_embeddings():
#     meta = json.load(open(META_FILE, "r", encoding="utf8"))
#     embeddings = np.load(EMB_FILE)
#     return meta, embeddings

# # Initialize model, messages, embeddings globally
# MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# MESSAGES = load_messages()

# if os.path.exists(EMB_FILE) and os.path.exists(META_FILE):
#     META, EMBEDDINGS = load_meta_and_embeddings()
# else:
#     # Compute embeddings if not precomputed
#     META = [{"user_name": m.get("user_name",""), "timestamp": m.get("timestamp","")} for m in MESSAGES]
#     EMBEDDINGS = MODEL.encode([m.get("message","") for m in MESSAGES], convert_to_numpy=True)
#     np.save(EMB_FILE, EMBEDDINGS)
#     with open(META_FILE, "w", encoding="utf8") as f:
#         json.dump(META, f, ensure_ascii=False, indent=2)

# # Semantic search utility
# def semantic_search(query: str, top_k=5) -> List[Dict]:
#     """
#     Return top_k most semantically similar messages to the query.
#     """
#     q_emb = MODEL.encode([query], convert_to_numpy=True)
#     sims = cosine_similarity(q_emb, EMBEDDINGS)[0]
#     top_idx = (-sims).argsort()[:top_k]
#     results = []
#     for idx in top_idx:
#         msg = MESSAGES[idx]
#         results.append({
#             "score": float(sims[idx]),
#             "user_name": msg.get("user_name"),
#             "timestamp": msg.get("timestamp"),
#             "message": msg.get("message")
#         })
#     return results

# # Fallback using semantic similarity for name-based queries
# def keyword_fallback(question: str, messages: List[Dict], top_k=5) -> str:
#     """
#     Enhanced fallback: uses embeddings to match partial names and context keywords.
#     """
#     # Extract capitalized tokens as potential names
#     name_tokens = re.findall(r"\b[A-Z][a-z]{1,20}\b", question)
#     if not name_tokens:
#         return ""

#     # Semantic match of tokens against user names
#     candidates = []
#     user_names = [m.get("user_name","") for m in messages]
#     for token in name_tokens:
#         token_emb = MODEL.encode([token], convert_to_numpy=True)
#         name_embs = MODEL.encode(user_names, convert_to_numpy=True)
#         sims = cosine_similarity(token_emb, name_embs)[0]
#         best_idx = np.argmax(sims)
#         if sims[best_idx] > 0.5:
#             candidates.append(user_names[best_idx])

#     if not candidates:
#         return ""

#     # Take the first candidate
#     person = candidates[0]

#     # Context keywords for different queries
#     query_contexts = {
#         "trip": ["trip", "travel", "flight"],
#         "cars": ["car", "cars"],
#         "favorites": ["favorite", "favourite", "restaurant", "restaurants"]
#     }

#     for context, keywords in query_contexts.items():
#         if any(word in question.lower() for word in keywords):
#             results = semantic_search(f"{person} {' '.join(keywords)}", top_k=top_k)
#             if results:
#                 best = results[0]
#                 return f"{best['user_name']}: {best['message']}"

#     # As last resort, semantic search with full question
#     results = semantic_search(question, top_k=top_k)
#     if results:
#         best = results[0]
#         return f"{best['user_name']}: {best['message']}"

#     return ""

# # Top-level answer function
# def answer(question: str) -> str:
#     """
#     Returns the most relevant message answering the question.
#     """
#     # 1) Try keyword fallback first
#     k = keyword_fallback(question, MESSAGES)
#     if k:
#         return k

#     # 2) Full semantic search
#     results = semantic_search(question, top_k=5)
#     if results:
#         best = results[0]
#         # Optional: threshold to avoid irrelevant answers
#         SIMILARITY_THRESHOLD = 0.5
#         if best["score"] < SIMILARITY_THRESHOLD:
#             return "Sorry — no relevant answer found."
#         return f"{best['user_name']}: {best['message']}"

#     return "Sorry — I couldn't find an answer in the dataset."


# # Example usage
# if __name__ == "__main__":
#     questions = [
#         "How many cars does Vikram Desai have?",
#         "When is Layla planning her trip to London?",
#         "What are Amira’s favorite restaurants?"
#     ]
#     for q in questions:
#         print(f"Q: {q}")
#         print(f"A: {answer(q)}\n")

# qa.py
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# Paths
DATA_FILE = "data/messages.json"
EMB_FILE = "data/embeddings.npy"
META_FILE = "data/meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load messages & embeddings
MESSAGES = json.load(open(DATA_FILE, "r", encoding="utf8"))

if os.path.exists(EMB_FILE) and os.path.exists(META_FILE):
    META = json.load(open(META_FILE, "r", encoding="utf8"))
    EMBEDDINGS = np.load(EMB_FILE)
else:
    raise FileNotFoundError("Run build_index.py first to generate embeddings.")

# Load embedding model
MODEL = SentenceTransformer(MODEL_NAME)

def semantic_search(question: str, top_k=5):
    """Return top-k semantically similar messages."""
    q_emb = MODEL.encode([question], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, EMBEDDINGS)[0]
    top_idx = (-sims).argsort()[:top_k]
    results = []
    for idx in top_idx:
        m = MESSAGES[idx]
        results.append({
            "score": float(sims[idx]),
            "user_name": m.get("user_name"),
            "timestamp": m.get("timestamp"),
            "message": m.get("message")
        })
    return results

def extract_numeric_answer(question: str, message: str) -> str:
    """Try to extract numbers (for cars, trips, etc.) from the message."""
    # Check for number words
    num_words = {"one":1, "two":2, "three":3, "four":4, "five":5}
    m = re.search(r"\b\d+\b", message)
    if m:
        return m.group(0)
    for word, val in num_words.items():
        if word in message.lower():
            return str(val)
    return message  # fallback: return full message

def answer(question: str, top_k=5):
    """Return the most relevant answer from the dataset using embeddings."""
    top_messages = semantic_search(question, top_k=top_k)
    if not top_messages:
        return "Sorry — no relevant answer found."

    best = top_messages[0]

    # Optional: lightweight extraction for questions about numbers
    if any(word in question.lower() for word in ["car", "cars", "number", "how many"]):
        numeric = extract_numeric_answer(question, best["message"])
        return f"{best['user_name']}: {numeric}"

    return f"{best['user_name']}: {best['message']}"

# Example console
if __name__ == "__main__":
    print("=== QA Console ===")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print("Answer:", answer(q), "\n")
