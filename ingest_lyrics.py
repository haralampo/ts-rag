from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize ChromaDB
client = chromadb.PersistentClient(path="./swift_vec_db")

# 2. Embedding model
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name
)

# 3. Reset collection
try:
    client.delete_collection("taylor_lyrics")
    print("Old collection deleted.")
except:
    pass

collection = client.create_collection(
    name="taylor_lyrics",
    embedding_function=embedding_func
)

# 4. Load all chunk CSVs
chunk_files = Path("data/chunks").glob("*_chunks.csv")

global_id = 0

for file in chunk_files:
    print(f"Ingesting {file.name}...")

    df = pd.read_csv(file)

    collection.add(
        documents=[
            f"Song: {row['song']}\nAlbum: {row['album']}\nSection: {row['section']}\nLyrics:\n{row['text']}"
            for _, row in df.iterrows()
        ],

        metadatas=[
            {
                "song": row["song"],
                "album": row["album"],
                "section": row["section"]
            }
            for _, row in df.iterrows()
        ],

        ids=[
            f"chunk_{global_id + i}"
            for i in range(len(df))
        ]
    )

    global_id += len(df)

print("Database populated!")