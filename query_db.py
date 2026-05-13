import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize ChromaDB
client = chromadb.PersistentClient(path="./swift_vec_db")

# 2. Choose embedding model
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# 3. Create collection (similar to SQL table)
# Stores text, embeddings (vectors), metadata
collection = client.get_or_create_collection(
    name="taylor_lyrics", 
    embedding_function=embedding_func
)

# 4. Query database
results = collection.query(
    query_texts=["I wanna break up with my boyfriend"],
    n_results=3
)

# 5. Pair doc, metadata, and distance of each result of FIRST (and only) QUERY
matches = zip(results['documents'][0], results['metadatas'][0], results['distances'][0])

for doc, metadata, distance in matches:
    print("=" * 50)
    print(f"Song: {metadata['song']}")
    print(f"Album: {metadata['album']}")
    print(f"Section: {metadata['section']}")
    print(f"Distance: {distance}")
    print()
    print(doc)
    print()