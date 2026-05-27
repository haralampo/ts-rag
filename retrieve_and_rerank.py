import json
from difflib import SequenceMatcher
from collections import defaultdict

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Configuration & Paths ---
DB_PATH = "./swift_vec_db"
COLLECTION_NAME = "taylor_lyrics"
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"

openai_client = OpenAI()


def too_similar(a, b, threshold=0.82):
    """
    Return True if two chunks are near-duplicates.
    Used to avoid repeated choruses/sections dominating results.
    """
    return SequenceMatcher(None, a, b).ratio() >= threshold


def classify_user_situation(query):
    """
    Convert the user's free-text situation into structured POV/timeline constraints.

    This gives the system an explicit target before retrieval instead of hoping
    vector similarity understands the user's role correctly.
    """
    prompt = f"""
Classify this user situation for matching Taylor Swift songs/lyrics.

User situation:
{query}

Focus on:
- what is concretely happening to the user
- whether the user caused the event or received the hurt
- whether the user is actively in pain or already healed
- what narrator roles should be preferred or avoided

Return JSON only in this exact format:
{{
  "user_role": "person dumped | betrayed partner | person leaving | regretful ex | conflicted narrator | hopeful partner | grieving person | anxious narrator | outsider/observer | other",
  "user_agency": "initiator | recipient | mutual | observer | unclear",
  "user_hurt_status": "hurt_by_other | hurting_other | self_blame | mutual_hurt | not_hurt | unclear",
  "timeline_state": "before_event | during_crisis | immediate_aftermath | unresolved_grief | healing | moved_on | reflective_closure | unclear",
  "emotional_state": ["emotion1", "emotion2", "emotion3"],
  "desired_song_pov": ["song narrator role that would fit"],
  "avoid_song_pov": ["song narrator role that would not fit"],
  "desired_timeline": ["timeline states that would fit"],
  "avoid_timeline": ["timeline states that would not fit"],
  "plain_english_summary": "one sentence summary of the user's situation"
}}
"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You classify user situations into concrete situation, narrative role, and timeline constraints. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


def expand_query(query, user_profile):
    """
    Generate a few semantic search queries while preserving concrete situation,
    POV, and timeline.

    Keep this narrow. Retrieval should find candidates close to the user's actual
    situation, not just broadly similar emotional songs.
    """
    prompt = f"""
User situation:
{query}

Structured user profile:
{json.dumps(user_profile, indent=2)}

Create 4 short vector-search queries for finding Taylor Swift songs/lyrics.

Rules:
- Preserve the user's concrete situation, POV, and primary timeline.
- Do not replace the user's situation with a neighboring situation just because the emotion is similar.
- Do not switch the user into the POV of the person who caused the hurt.
- Focus on the user's PRIMARY timeline first.
- Only include secondary timeline wording if it still preserves the user's concrete situation.
- Include both literal and emotional wording, but keep the same narrator role.

Return JSON only:
{{
  "queries": ["query 1", "query 2", "query 3", "query 4"]
}}
"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You create concise semantic search queries that preserve concrete situation, POV, and timeline. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.25
    )

    data = json.loads(response.choices[0].message.content)
    queries = data.get("queries", [])

    # Include original query first.
    return [query] + [
        q for q in queries
        if isinstance(q, str) and q.strip() and q != query
    ]


def query_one(collection, query, fetch_k=40):
    """
    Run one vector query and normalize Chroma's nested response into dictionaries.
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
    query,
    user_profile,
    fetch_k_per_query=40,
    max_candidates=35,
    max_chunks_per_song=3
):
    """
    Retrieve a broad candidate pool.

    Important:
    - Do not dedupe to one result per song too early.
    - Keep a few sections per song so the reranker can decide which section fits best.
    """
    expanded_queries = expand_query(query, user_profile)
    raw_matches = []

    print("User profile:")
    print(json.dumps(user_profile, indent=2))

    print("\nExpanded queries:")
    for q in expanded_queries:
        print(f"- {q}")

    for q in expanded_queries:
        raw_matches.extend(query_one(collection, q, fetch_k=fetch_k_per_query))

    # Lower Chroma distance is better.
    raw_matches.sort(key=lambda m: m["distance"])

    filtered = []
    seen_docs = set()
    chunks_per_song = defaultdict(int)

    for match in raw_matches:
        doc = match["document"]
        meta = match["metadata"]
        song = meta.get("song", "")

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

    return filtered


def rerank_matches(query, user_profile, matches, final_k=5):
    """
    Ask the LLM to judge candidates using concrete situation, POV, and timeline.

    Chroma gives possible matches.
    The LLM decides whether they actually fit.
    """
    candidates = []

    for i, match in enumerate(matches):
        meta = match["metadata"]

        candidates.append({
            "index": i,
            "song": meta.get("song", ""),
            "album": meta.get("album", ""),
            "section": meta.get("section", ""),

            "metadata": {
                "speaker_role": meta.get("speaker_role", ""),
                "narrator_agency": meta.get("narrator_agency", ""),
                "narrator_hurt_status": meta.get("narrator_hurt_status", ""),
                "relationship_stage": meta.get("relationship_stage", ""),
                "timeline_state": meta.get("timeline_state", ""),
                "perspective": meta.get("perspective", ""),
                "good_for_user_roles": meta.get("good_for_user_roles", ""),
                "bad_for_user_roles": meta.get("bad_for_user_roles", ""),
                "summary": meta.get("summary", ""),
                "themes": meta.get("themes", ""),
                "moods": meta.get("moods", ""),
                "situations": meta.get("situations", "")
            },

            "content": match["document"],
            "distance": match["distance"],
            "source_query": match["source_query"]
        })

    prompt = f"""
User situation:
{query}

Structured user profile:
{json.dumps(user_profile, indent=2)}

You are ranking Taylor Swift lyric candidates.

Goal:
Pick lyrics where the narrator's concrete situation, POV, causal role, emotional timeline, and lyric evidence match the user.

Use metadata as a helpful signal, but verify it against the actual lyric/content text. Do not blindly trust metadata if the lyric section itself is vague or does not support the match.

CRITICAL GATES:

1. CONCRETE SITUATION GATE
- Identify the user's concrete situation, not just the general emotion.
- Prefer candidates that match what is actually happening to the user.
- Do not substitute a neighboring situation just because it has a similar mood.
- Examples of neighboring-but-different situations:
  - being left vs being the one leaving
  - being hurt by someone else's action vs regretting your own mistake
  - grieving a death/loss vs missing an ex
  - public shame vs private heartbreak
  - anxiety/self-doubt vs romantic rejection
  - revenge/anger vs empowerment/healing
  - nostalgia/reflection vs active crisis
- If the candidate matches the emotion but not the concrete situation, cap score at 8.
- If the candidate is only generally sad, angry, romantic, nostalgic, or emotional, cap score at 6.

2. CAUSAL ROLE / POV GATE
- Identify whether the user is the initiator, recipient, observer, or conflicted participant.
- Prefer narrators with the same causal role as the user.
- If the user is hurt by someone else, prefer narrators who are also hurt by someone else.
- Penalize narrators who caused the hurt, inflicted the damage, abandoned someone, betrayed someone, or are reflecting on their own mistake when the user is the recipient of harm.
- If the candidate's narrator agency directly conflicts with the user's agency, cap score at 4.

3. TIMELINE GATE
- Identify where the user is emotionally in time: before the event, during the crisis, immediate aftermath, unresolved grief, healing, moved on, or reflective closure.
- Prefer candidates in the same timeline phase.
- If the user is actively in pain, freshly hurt, shocked, or unresolved, prefer active pain / during-crisis / immediate aftermath.
- Unresolved grief is allowed, but it is not the same as immediate aftermath.
- If the user timeline is immediate_aftermath:
  - immediate_aftermath or during_crisis may score 9-10
  - unresolved_grief may score at most 8 unless the lyric section clearly shows fresh shock, sudden abandonment, or active crisis
  - reflective nostalgia may score at most 6
  - healing, moved_on, or reflective_closure may score at most 5
- If the candidate is clearly healed/moved-on while the user is still in crisis, cap score at 5.

4. LYRIC EVIDENCE GATE
- The actual lyric section must visibly support the claimed match.
- Score based on the displayed lyric section, not just the whole song profile.
- If the metadata says the match is strong but the lyric section itself is vague, cap score at 7.
- If the lyric section sounds like general reflection rather than the user's current situation, cap score at 6.
- A strong song profile cannot rescue a weak or vague selected lyric section.

5. SCORE CALIBRATION GATE
- Be strict with high scores.
- A score of 10 is rare.
- Only give 10 if the lyric section directly matches the user's concrete situation, POV, causal role, and timeline.
- If multiple candidates are strong, do not give all of them 10. Rank by exactness.
- A candidate can be emotionally resonant and still only deserve 7 or 8 if the concrete situation differs.

6. RESONANCE GATE
- Reward lyrics that would feel emotionally satisfying, validating, or painfully accurate to the user.
- Do not reward shallow keyword overlap.
- A result should feel like: "Yes, this is exactly what I'm going through," not just "this has a similar emotion."

Scoring:
- 10: Rare. Exact concrete situation, same POV, same causal role, same emotional timeline, and the lyric section itself strongly proves the match.
- 9: Very strong match with only tiny differences.
- 7-8: Strong emotional fit, but less exact, less direct, or adjacent situation.
- 5-6: General thematic/mood fit, but situation, POV, or timeline is imperfect.
- 1-4: Wrong POV, wrong agency, wrong emotional timeline, or mostly keyword-related.

Return JSON only:
{{
  "rankings": [
    {{
      "index": 0,
      "score": 10,
      "match_type": "exact_situation_match | close_emotional_match | general_theme_match | poor_match",
      "situation_analysis": "Identify the user's concrete situation and whether the candidate matches it directly or only emotionally.",
      "pov_timeline_analysis": "Explain narrator agency, hurt status, timeline fit/mismatch, and whether the lyric section itself proves the match.",
      "reason": "Short user-facing explanation of why this matches."
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
                "content": "You are a strict concrete-situation, POV, and timeline-aware lyric match judge. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    data = json.loads(response.choices[0].message.content)
    rankings = sorted(data["rankings"], key=lambda x: x.get("score", 0), reverse=True)

    reranked = []

    for item in rankings:
        idx = item.get("index")

        if not isinstance(idx, int) or idx < 0 or idx >= len(matches):
            continue

        match = matches[idx].copy()
        match["rerank_score"] = item.get("score", 0)
        match["match_type"] = item.get("match_type", "")
        match["situation_analysis"] = item.get("situation_analysis", "")
        match["pov_timeline_analysis"] = item.get("pov_timeline_analysis", "")
        match["reason"] = item.get("reason", "")

        reranked.append(match)

    return enforce_final_diversity(reranked, final_k=final_k)


def enforce_final_diversity(reranked, final_k=5):
    """
    After reranking, keep the best results while avoiding repeated songs
    and near-identical chunks.
    """
    final = []
    seen_songs = set()

    for match in reranked:
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

    return final


def print_matches(matches):
    for match in matches:
        meta = match["metadata"]

        print("=" * 50)
        print(f"Score: {match.get('rerank_score', '')}/10")
        print(f"Distance: {match.get('distance', '')}")
        print(f"Song: {meta.get('song', '')}")
        print(f"Album: {meta.get('album', '')}")
        print(f"Section: {meta.get('section', '')}")
        print(f"Speaker Role: {meta.get('speaker_role', '')}")
        print(f"Agency: {meta.get('narrator_agency', '')}")
        print(f"Hurt Status: {meta.get('narrator_hurt_status', '')}")
        print(f"Timeline: {meta.get('timeline_state', '')}")
        print(f"Match Type: {match.get('match_type', '')}")
        print(f"Situation Analysis: {match.get('situation_analysis', '')}")
        print(f"POV/Timeline Analysis: {match.get('pov_timeline_analysis', '')}")
        print(f"Why: {match.get('reason', '')}")
        print()
        print(match["document"])
        print()


def run_lyric_search(user_query, num_results=5):
    """
    Main function for future UI use.
    """
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    user_profile = classify_user_situation(user_query)

    candidates = retrieve_candidates(
        collection=collection,
        query=user_query,
        user_profile=user_profile,
        fetch_k_per_query=40,
        max_candidates=35,
        max_chunks_per_song=3
    )

    if not candidates:
        return []

    reranked = rerank_matches(
        query=user_query,
        user_profile=user_profile,
        matches=candidates,
        final_k=num_results
    )

    return reranked


def main():
    query = "I just got dumped and it was a major blindside"

    matches = run_lyric_search(query, num_results=5)

    print(f"\nFinal matches: {len(matches)}\n")
    print_matches(matches)


if __name__ == "__main__":
    main()