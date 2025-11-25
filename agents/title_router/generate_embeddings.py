# title_router/generate_embeddings.py

import json
from openai import OpenAI

client = OpenAI()

# 1) Load titles
with open("category_titles.json", "r") as f:
    titles = json.load(f)

# 2) Create embeddings
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=titles
)

embeddings = [item.embedding for item in response.data]

# 3) Save embeddings
with open("title_embeddings.json", "w") as f:
    json.dump(embeddings, f)

print("✅ OpenAI: title embeddings generated!")
