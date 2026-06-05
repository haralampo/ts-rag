import json
import re
from difflib import SequenceMatcher
from collections import defaultdict

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


# --- Configuration ---
DB_PATH = "./swift_vec_db_public"
COLLECTION_NAME = "taylor_lyrics"
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"

# CHANGE:
# Keep debug logs off in the public demo.
DEBUG = False

openai_client = OpenAI()


# -----------------------------
# Collection loading
# -----------------------------

def load_collection():
    """
    Load the Chroma collection with the same embedding model used at ingestion.

    CHANGE:
    This is now separate from run_lyric_search so Streamlit can cache it.
    Re-loading the embedding model on every search is expensive.
    """
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    return collection


# -----------------------------
# Utility helpers
# -----------------------------

def too_similar(a, b, threshold=0.84):
    """
    Return True if two stored chunks are near-duplicates.
    """
    return SequenceMatcher(None, a, b).ratio() >= threshold


def safe_json_loads(text, fallback):
    """
    Parse model JSON safely. If parsing fails, return a fallback instead of
    crashing the entire search.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def clean_query(q):
    """
    Convert search-style generated queries into natural semantic retrieval queries.
    """
    if not isinstance(q, str):
        return ""

    q = q.strip()

    q = re.sub(
        r"^(taylor swift\s+)?(a\s+|the\s+)?"
        r"(song|songs|lyric|lyrics)\s+"
        r"(that\s+)?"
        r"(about|capture|captures|express|expresses|reflect|reflects|show|shows|assert|asserts|convey|conveys)\s+",
        "",
        q,
        flags=re.IGNORECASE
    )

    q = re.sub(
        r"^(the\s+)?(feeling|emotion|experience|situation)\s+of\s+",
        "",
        q,
        flags=re.IGNORECASE
    )

    q = q.strip(" .,:;-")
    return q


def debug_print(*args, **kwargs):
    """
    Print only when DEBUG is enabled.
    """
    if DEBUG:
        print(*args, **kwargs)


# -----------------------------
# Search planning
# -----------------------------

def build_search_plan(user_query):
    """
    Create a concise search plan.

    CHANGE:
    The search plan stays because it is core to the project.
    But it now asks for 3 generated queries instead of 4, and the final search
    uses only 3 total queries: the raw user query plus up to 2 generated ones.
    """
    prompt = f"""
User situation:
{user_query}

Create a concise search plan for finding song sections that match this situation.

Focus on:
- what is actually happening
- the user's emotional stance
- the user's POV
- the user's timeline/phase
- what the matching narrator should feel, want, or need
- what kinds of song sections would feel validating
- what kinds of song sections would be misleading

Important:
- Do not assume the situation is romantic unless the user says so.
- Do not collapse all sadness into heartbreak.
- Do not collapse all anger into revenge.
- Do not collapse all moving on into healing if the user sounds detached, done, or annoyed.
- Preserve the user's POV. Do not switch into the perspective of the person causing the problem.

Query rules:
- Write retrieval queries as diary-like situation descriptions.
- Prefer first-person wording when the user is speaking from their own POV.
- Do not include "Taylor Swift", "song", "songs", "lyric", or "lyrics".
- Do not ask a question unless the user's situation is naturally a question.
- Do not turn the query into a web search phrase.
- Good: "I am done explaining myself and want this person to accept it"
- Bad: "Taylor Swift lyrics about setting boundaries"

Return JSON only:
{{
  "situation_summary": "one sentence summary of the concrete situation",
  "target_narrator_state": {{
    "emotional_state": "what the matching narrator should feel",
    "agency_level": "passive | conflicted | setting_boundaries | taking_action | detached | reflective",
    "timeline_phase": "before_event | during_event | immediate_aftermath | unresolved | healing | moved_on | reflective_closure | uncertain",
    "speaker_role": "the role/perspective the narrator should have"
  }},
  "good_match_signals": ["signal1", "signal2", "signal3"],
  "avoid_match_signals": ["signal1", "signal2", "signal3"],
  "queries": [
    "first-person natural situation query",
    "first-person emotional state query",
    "first-person timeline or agency query"
  ]
}}
"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You create concise retrieval plans. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    plan = safe_json_loads(
        response.choices[0].message.content,
        fallback={
            "situation_summary": user_query,
            "target_narrator_state": {
                "emotional_state": "",
                "agency_level": "",
                "timeline_phase": "",
                "speaker_role": ""
            },
            "good_match_signals": [],
            "avoid_match_signals": [],
            "queries": []
        }
    )

    queries = [user_query]

    for q in plan.get("queries", []):
        cleaned = clean_query(q)
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    target_narrator_state = plan.get("target_narrator_state", {})

    return {
        "situation_summary": plan.get("situation_summary", user_query),
        "target_narrator_state": {
            "emotional_state": target_narrator_state.get("emotional_state", ""),
            "agency_level": target_narrator_state.get("agency_level", ""),
            "timeline_phase": target_narrator_state.get("timeline_phase", ""),
            "speaker_role": target_narrator_state.get("speaker_role", "")
        },
        "good_match_signals": plan.get("good_match_signals", [])[:3],
        "avoid_match_signals": plan.get("avoid_match_signals", [])[:3],

        # CHANGE:
        # Use only 3 total retrieval queries.
        # This keeps query expansion but reduces retrieval and reranking work.
        "queries": queries[:3]
    }


# -----------------------------
# Retrieval
# -----------------------------

def query_one(collection, query, fetch_k=25):
    """
    Run one Chroma query and normalize the response.

    CHANGE:
    fetch_k default lowered from 35 to 25.
    """
    results = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        matches.append({
            "document": doc,
            "metadata": meta,
            "distance": distance,
            "source_query": query
        })

    return matches


def retrieve_candidates(
    collection,
    search_plan,
    fetch_k_per_query=25,
    max_candidates=15,
    max_chunks_per_song=2
):
    """
    Retrieve candidates from Chroma using the search plan queries.

    CHANGE:
    - fetch_k_per_query lowered to 25
    - max_candidates lowered to 15
    - max_chunks_per_song stays at 2
    """
    raw_matches = []

    debug_print("Search plan:")
    debug_print(json.dumps(search_plan, indent=2))

    debug_print("\nSearch queries:")
    for query in search_plan["queries"]:
        debug_print(f"- {query}")
        raw_matches.extend(query_one(collection, query, fetch_k=fetch_k_per_query))

    raw_matches.sort(key=lambda m: m["distance"])

    filtered = []
    seen_docs = set()
    chunks_per_song = defaultdict(int)

    for match in raw_matches:
        doc = match["document"]
        song = match["metadata"].get("song", "")

        if doc in seen_docs:
            continue

        if chunks_per_song[song] >= max_chunks_per_song:
            continue

        if any(too_similar(doc, kept["document"]) for kept in filtered):
            continue

        filtered.append(match)
        seen_docs.add(doc)
        chunks_per_song[song] += 1

        if len(filtered) >= max_candidates:
            break

    debug_print(f"\nCandidates sent to reranker: {len(filtered)}")
    return filtered


# -----------------------------
# Reranking
# -----------------------------

def build_rerank_candidates(matches):
    """
    Build a compact candidate list for the LLM reranker.

    CHANGE:
    The old version sent larger repeated profile/document text.
    This version sends only the metadata fields the reranker needs.
    """
    candidates = []

    for i, match in enumerate(matches):
        meta = match["metadata"]

        candidates.append({
            "index": i,
            "song": meta.get("song", ""),
            "album": meta.get("album", ""),
            "section": meta.get("section", ""),
            "themes": meta.get("themes", ""),
            "moods": meta.get("moods", ""),
            "situations": meta.get("situations", ""),
            "perspective": meta.get("perspective", ""),
            "speaker_role": meta.get("speaker_role", ""),
            "narrator_agency": meta.get("narrator_agency", ""),
            "narrator_hurt_status": meta.get("narrator_hurt_status", ""),
            "relationship_stage": meta.get("relationship_stage", ""),
            "timeline_state": meta.get("timeline_state", ""),
            "summary": meta.get("summary", ""),
            "distance": round(match["distance"], 4),
            "source_query": match["source_query"]
        })

    return candidates


def rerank_matches(user_query, search_plan, matches, final_k=5, min_score=7):
    """
    Rerank candidates by concrete situation fit, narrator state, timeline,
    speaker role, and section metadata.

    CHANGE:
    The public version does not include raw lyric text, so the prompt now judges
    from song-section/profile metadata instead of pretending lyric text exists.
    """
    candidates = build_rerank_candidates(matches)

    prompt = f"""
User situation:
{user_query}

Search plan:
{json.dumps(search_plan, indent=2)}

You are ranking song-section candidates for a public Taylor Swift match demo.

Pick sections that match the user's actual situation, POV, emotional stance, narrator state, and timeline.

The public version does not include raw lyric text. Judge only from the provided song-section metadata:
- themes
- moods
- situations
- perspective
- speaker_role
- narrator_agency
- narrator_hurt_status
- relationship_stage
- timeline_state
- summary

Evaluation rules:

1. Concrete situation fit
- Does the candidate match what is actually happening to the user?
- Do not treat neighboring situations as identical.
- Examples of different situations:
  - being over someone vs still missing them
  - setting a boundary vs wanting reconciliation
  - public shame vs private heartbreak
  - grief/loss vs romantic rejection
  - revenge vs empowerment
  - anxiety/self-doubt vs romantic rejection
  - active crisis vs reflective closure

2. Narrator state alignment
- Compare the user's target narrator state against the candidate's narrator state.
- The candidate should match emotional state, agency level, timeline phase, and speaker role.
- Penalize wrong-state matches even when the general topic is similar.
- Wrong-state matches should usually score 5 or below.

3. Timeline alignment
- Before, during, immediate aftermath, unresolved pain, healing, moved-on detachment, and reflective closure are different phases.
- A candidate can share the same topic but still be wrong if it speaks from the wrong phase.

4. Score calibration
- 10 is rare.
- Prefer 2-3 strong matches over filling the list with weak ones.
- Do not score a candidate 7+ if the narrator state conflicts with the user state.
- Do not score a candidate 7+ just because it has related keywords.
- Do not use outside knowledge of the full song.

Scoring:
- 10: Exact concrete situation, same narrator state, same timeline phase.
- 9: Very strong match with tiny differences.
- 7-8: Good match; narrator state and timeline are right.
- 6: Usable near-match; tone or theme fits, but situation is not exact.
- 5: General mood/theme match, but narrator state, role, or timeline is noticeably off.
- 1-4: Wrong narrator state, wrong speaker role, wrong timeline, or mostly keyword overlap.

Return JSON only.
Return one ranking entry for every candidate.

Format:
{{
  "rankings": [
    {{
      "index": 0,
      "score": 8,
      "match_type": "exact_situation_match | close_state_match | usable_near_match | general_theme_match | poor_match",
      "section_is_about": "one sentence describing what this section/song profile seems to be about",
      "narrator_state": "what the narrator seems to feel/want/need",
      "state_alignment": "same_state | close_state | partial_state | wrong_state",
      "analysis": "briefly explain situation fit, narrator state fit, and timeline fit",
      "reason": "short user-facing explanation of why this section matches"
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
                "content": "You are a strict but practical song-section match judge. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    data = safe_json_loads(response.choices[0].message.content, fallback={"rankings": []})

    rankings = sorted(
        data.get("rankings", []),
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    reranked = []

    for item in rankings:
        idx = item.get("index")

        if not isinstance(idx, int) or idx < 0 or idx >= len(matches):
            continue

        match = matches[idx].copy()
        match["rerank_score"] = item.get("score", 0)
        match["match_type"] = item.get("match_type", "")
        match["section_is_about"] = item.get("section_is_about", "")
        match["narrator_state"] = item.get("narrator_state", "")
        match["state_alignment"] = item.get("state_alignment", "")
        match["analysis"] = item.get("analysis", "")
        match["reason"] = item.get("reason", "")

        reranked.append(match)

    print_rerank_summary(reranked)

    return select_final_matches(
        reranked=reranked,
        final_k=final_k,
        min_score=min_score
    )


def print_rerank_summary(reranked, top_n=8):
    """
    Print highest-scored candidates before final filtering.
    Disabled unless DEBUG = True.
    """
    if not DEBUG:
        return

    if not reranked:
        print("\nReranker returned no usable rankings.")
        return

    print("\nTop reranked candidates before final filtering:")

    for match in reranked[:top_n]:
        meta = match["metadata"]

        print(
            f"- {match.get('rerank_score', 0)}/10 | "
            f"{meta.get('song', '')} | {meta.get('section', '')} | "
            f"{match.get('match_type', '')} | "
            f"{match.get('state_alignment', '')}"
        )


def select_final_matches(reranked, final_k=5, min_score=7):
    """
    Select confident, diverse matches.

    Main behavior:
    - require min_score
    - avoid duplicate songs
    - avoid near-duplicate sections
    - avoid wrong-state matches

    Fallback behavior:
    - if nothing passes min_score, return the best diverse near-matches with
      scores >= 6 and label them low-confidence
    """
    final = []
    seen_songs = set()

    for match in reranked:
        if match.get("rerank_score", 0) < min_score:
            continue

        if match.get("state_alignment") == "wrong_state":
            continue

        song = match["metadata"].get("song", "")
        doc = match["document"]

        if song in seen_songs:
            continue

        if any(too_similar(doc, kept["document"]) for kept in final):
            continue

        final.append(match)
        seen_songs.add(song)

        if len(final) >= final_k:
            break

    if final:
        return final

    fallback = []
    seen_songs = set()

    for match in reranked:
        if match.get("rerank_score", 0) < 6:
            continue

        if match.get("state_alignment") == "wrong_state":
            continue

        song = match["metadata"].get("song", "")
        doc = match["document"]

        if song in seen_songs:
            continue

        if any(too_similar(doc, kept["document"]) for kept in fallback):
            continue

        match = match.copy()
        match["match_type"] = match.get("match_type") or "usable_near_match"
        match["low_confidence"] = True

        fallback.append(match)
        seen_songs.add(song)

        if len(fallback) >= final_k:
            break

    return fallback


# -----------------------------
# Output / entry point
# -----------------------------

def print_matches(matches):
    """
    Local CLI helper.
    """
    for match in matches:
        meta = match["metadata"]
        confidence_note = "LOW CONFIDENCE NEAR-MATCH" if match.get("low_confidence") else ""

        print("=" * 50)

        if confidence_note:
            print(confidence_note)

        print(f"Score: {match.get('rerank_score', '')}/10")
        print(f"Distance: {match.get('distance', '')}")
        print(f"Song: {meta.get('song', '')}")
        print(f"Album: {meta.get('album', '')}")
        print(f"Section: {meta.get('section', '')}")
        print(f"Match Type: {match.get('match_type', '')}")
        print(f"Section is about: {match.get('section_is_about', '')}")
        print(f"Narrator State: {match.get('narrator_state', '')}")
        print(f"State Alignment: {match.get('state_alignment', '')}")
        print(f"Analysis: {match.get('analysis', '')}")
        print(f"Why: {match.get('reason', '')}")
        print()


def run_lyric_search(user_query, num_results=5, min_score=7, collection=None):
    """
    Main entry point for CLI and Streamlit.

    CHANGE:
    Accepts optional cached collection.
    Streamlit passes this in so the collection/model are not reloaded per search.
    """
    if collection is None:
        collection = load_collection()

    search_plan = build_search_plan(user_query)

    candidates = retrieve_candidates(
        collection=collection,
        search_plan=search_plan,
        fetch_k_per_query=25,
        max_candidates=15,
        max_chunks_per_song=2
    )

    if not candidates:
        return []

    return rerank_matches(
        user_query=user_query,
        search_plan=search_plan,
        matches=candidates,
        final_k=num_results,
        min_score=min_score
    )


def main():
    query = "My boyfriend wants to break up, but I don't want to"

    matches = run_lyric_search(
        user_query=query,
        num_results=5,
        min_score=7
    )

    print(f"\nFinal matches: {len(matches)}\n")
    print_matches(matches)


if __name__ == "__main__":
    main()