import json
from difflib import SequenceMatcher

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Configuration & Paths ---
DB_PATH = "./swift_vec_db"           # Directory where ChromaDB reads local vector data files
COLLECTION_NAME = "taylor_lyrics"    # Target database collection to execute queries against
MODEL_NAME = "all-MiniLM-L6-v2"      # Transformer model matching the original ingestion configuration
OPENAI_MODEL = "gpt-4o-mini"         # OpenAI model optimized for query generation and metadata reranking

openai_client = OpenAI()


# Use Gestalt Pattern Matching to calculate string similarity ratios and eliminate near-duplicate text
def too_similar(a, b, threshold=0.82):
    return SequenceMatcher(None, a, b).ratio() >= threshold


# Leverage OpenAI to generate a compressed semantic search matrix from a single user input
def expand_query(query):
    # FIX: Reduced query expansion to 3 variations to prevent lower-intent noise at the top of the funnel
    prompt = f"""
User situation:
{query}

Create 3 short search queries that capture different emotional interpretations
of this situation for matching Taylor Swift songs/lyrics.

Avoid just repeating the same words.

Return JSON only:
{{
  "queries": ["query 1", "query 2", "query 3"]
}}
"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You create concise semantic search queries. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    data = json.loads(response.choices[0].message.content)
    queries = data.get("queries", [])

    return [query] + [q for q in queries if q and q != query]


# Execute vector similarity searches while enforcing diversity constraints to prevent track clustering
def query_diverse(collection, query, final_k=8, fetch_k=25):
    results = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    filtered = []
    seen_songs = set()

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        song_key = meta["song"]

        if song_key in seen_songs:
            continue

        if any(too_similar(doc, kept["document"]) for kept in filtered):
            continue

        filtered.append({
            "document": doc,
            "metadata": meta,
            "distance": distance,
            "source_query": query
        })

        seen_songs.add(song_key)

        if len(filtered) == final_k:
            break

    return filtered


# Execute parallel sweeps using query matrix, then consolidate unique matches
def retrieve_candidates(collection, query, per_query_k=8, fetch_k=25):
    expanded_queries = expand_query(query)
    all_matches = []

    print("Expanded queries:")
    for q in expanded_queries:
        print(f"- {q}")

    for q in expanded_queries:
        all_matches.extend(
            query_diverse(
                collection,
                query=q,
                final_k=per_query_k,
                fetch_k=fetch_k
            )
        )

    unique = []
    seen_docs = set()
    seen_songs = set()

    for match in all_matches:
        meta = match["metadata"]
        doc = match["document"]
        song_key = meta["song"]

        if doc in seen_docs or song_key in seen_songs:
            continue

        if any(too_similar(doc, kept["document"]) for kept in unique):
            continue

        unique.append(match)
        seen_docs.add(doc)
        seen_songs.add(song_key)

    return unique


# Use LLM to evaluate narrative perspective and score candidate relevance
def rerank_matches(query, matches, final_k=5):
    candidates = []

    # Map database metadata structures into a clean, minimal context packet
    for i, match in enumerate(matches):
        meta = match["metadata"]
        
        candidates.append({
            "index": i,
            "song": meta["song"],
            "album": meta["album"],
            "section": meta["section"],
            # match["document"] ALREADY contains both the Song Profile text AND the lyrics!
            "content": match["document"] 
        })

    # CLEAN, UNIVERSAL PROMPT: No specific song names, no relationship-only jargon
    prompt = f"""
User Situation / Emotional State:
{query}

You are an objective text analyst. Score each candidate below from 1 to 10 based strictly on how perfectly the narrator's emotional perspective, causal role, and timeline align with the user.

CRITICAL EVALUATION GATES:
1. THE CAUSAL ROLE GATE: Identify the flow of action. Is the user the initiator/cause of their situation, or are they the passive recipient/target of someone else's action?
   - Look at the candidate's content. If there is a direct mismatch in agency (e.g., the user is the recipient of an action, but the narrator explicitly describes being the initiator who caused, chose, or drove the event), you MUST cap the score at a MAXIMUM of 4.

2. THE EMOTIONAL TIMELINE GATE: Identify the trajectory of the text. Is the user experiencing an active, raw, un-healed scenario?
   - Look at the candidate's content. If the narrator explicitly frames the event as a closed past chapter, declares that they have fully recovered, or details a subsequent successful life phase beyond the crisis, you MUST cap the score at a MAXIMUM of 5.
   - Note: Do NOT penalize standard past-tense storytelling if the core text lines show the narrator is still actively trapped in, mourning, or reeling from the unresolved weight of the event.

SCORING MATRIX:
- 9 to 10: Seamless alignment. The narrator is experiencing the exact same causal role, emotional intensity, and unresolved phase as the user.
- 6 to 8: Strong thematic match, but with slight situational variations or minor narrative differences.
- 1 to 5: Failed either the Causal Role Gate or the Emotional Timeline Gate.

Return JSON format only:
{{
  "rankings": [
    {{
      "index": 0,
      "narrator_perspective_analysis": "Identify if narrator is initiator vs recipient, and if they are in an active vs resolved timeline phase. Note any gate violations.",
      "score": 10,
      "reason": "Objective explanation based strictly on text indicators."
    }}
  ]
}}

Candidates:
{json.dumps(candidates, indent=2)}
"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a literal, objective analyst evaluating raw text perspectives and timelines without relying on external tags or labels."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    data = json.loads(response.choices[0].message.content)
    rankings = sorted(data["rankings"], key=lambda x: x["score"], reverse=True)

    reranked = []
    for item in rankings[:final_k]:
        idx = item["index"]
        if idx < 0 or idx >= len(matches):
            continue

        match = matches[idx].copy()
        match["rerank_score"] = item["score"]
        match["reason"] = item["reason"]
        match["pov_analysis"] = item.get("narrator_perspective_analysis", "N/A")
        reranked.append(match)

    return reranked

# Parse, layout, and print the final evaluation structures
def print_matches(matches):
    for match in matches:
        meta = match["metadata"]

        print("=" * 50)
        print(f"Score: {match['rerank_score']}/10")
        print(f"Distance: {match['distance']}")
        print(f"Song: {meta['song']}")
        print(f"Album: {meta['album']}")
        print(f"Section: {meta['section']}")
        print(f"Narrator POV: {match.get('pov_analysis', '')}")
        print(f"Why: {match['reason']}\n")
        print(match["document"])


def main():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    query = "I just got dumped and it was a major blindside"

    candidates = retrieve_candidates(
        collection,
        query=query,
        per_query_k=8,
        fetch_k=30
    )

    print(f"\nRetrieved {len(candidates)} unique candidates.\n")

    reranked = rerank_matches(
        query=query,
        matches=candidates,
        final_k=5
    )

    print_matches(reranked)


if __name__ == "__main__":
    main()