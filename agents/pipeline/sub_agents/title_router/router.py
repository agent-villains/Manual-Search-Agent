import json
import numpy as np
import os
from openai import OpenAI

client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "category_titles.json"), "r") as f:
    CATEGORY_TITLES = json.load(f)

with open(os.path.join(BASE_DIR, "title_embeddings.json"), "r") as f:
    title_embeddings = np.array(json.load(f))


def embed_question(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return np.array(response.data[0].embedding)


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def route_by_title(question: str):
    q_emb = embed_question(question)
    sims = [cosine(q_emb, t) for t in title_embeddings]
    idx = int(np.argmax(sims))
    return {
        "best_title": CATEGORY_TITLES[idx],
        "score": float(sims[idx])
    }
