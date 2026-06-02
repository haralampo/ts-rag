import json
import re
from difflib import SequenceMatcher
from collections import defaultdict
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Configuration ---
DB_PATH = "./swift_vec_db"
COLLECTION_NAME = "taylor_lyrics"
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"

openai_client = OpenAI()


# -----------------------------
# Logging
# -----------------------------

def save_search_log(
    user_query,
    matches,
    jsonl_filename="search_logs.jsonl",
    pretty_filename="search_logs_pretty.txt"
):
    """
    Save each search run in two formats:

    1. JSONL file:
       - one compact JSON object per line
       - best for pandas / later analysis

    2. Pretty text file:
       - readable formatting
       - best for manually reviewing outputs
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_query": user_query,
        "num_matches": len(matches),
        "matches": []
    }

    for match in matches:
        meta = match.get("metadata", {})
        profile_text, lyrics_text = split_document(match.get("document", ""))

        log_entry["matches"].append({
            "song": meta.get("song", ""),
            "album": meta.get("album", ""),
            "section": meta.get("section", ""),
            "score": match.get("rerank_score", ""),
            "distance": match.get("distance", ""),
            "match_type": match.get("match_type", ""),
            "state_alignment": match.get("state_alignment", ""),
            "lyric_is_about": match.get("lyric_is_about", ""),
            "narrator_state": match.get("narrator_state", ""),
            "meaningful_lyric_words": meaningful_word_count(lyrics_text),
            "analysis": match.get("analysis", ""),
            "reason": match.get("reason", ""),
            "lyrics": lyrics_text,
            "profile_text": profile_text,
            "source_query": match.get("source_query", "")
        })

    # 1. Compact JSONL version for data analysis
    with open(jsonl_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # 2. Pretty readable version for manual review
    with open(pretty_filename, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {log_entry['timestamp']}\n")
        f.write(f"User Query: {log_entry['user_query']}\n")
        f.write(f"Number of Matches: {log_entry['num_matches']}\n")
        f.write("=" * 80 + "\n\n")

        for i, match in enumerate(log_entry["matches"], start=1):
            f.write(f"Match {i}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Song: {match['song']}\n")
            f.write(f"Album: {match['album']}\n")
            f.write(f"Section: {match['section']}\n")
            f.write(f"Score: {match['score']}/10\n")
            f.write(f"Distance: {match['distance']}\n")
            f.write(f"Match Type: {match['match_type']}\n")
            f.write(f"State Alignment: {match['state_alignment']}\n")
            f.write(f"Meaningful Lyric Words: {match['meaningful_lyric_words']}\n")
            f.write(f"Lyric Is About: {match['lyric_is_about']}\n")
            f.write(f"Narrator State: {match['narrator_state']}\n")
            f.write(f"Source Query: {match['source_query']}\n\n")

            f.write("Analysis:\n")
            f.write(f"{match['analysis']}\n\n")

            f.write("Why:\n")
            f.write(f"{match['reason']}\n\n")

            f.write("Lyrics:\n")
            f.write(f"{match['lyrics']}\n\n")

        f.write("\n\n")


# -----------------------------
# Utility helpers
# -----------------------------

def too_similar(a, b, threshold=0.84):
    """
    Return True if two text chunks are near-duplicates.

    This removes obvious repeated sections without being so aggressive that
    different sections with similar refrains get filtered out.
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

    The vector DB contains lyric chunks, not web pages, so retrieval works better
    with situation/emotion language than with phrases like:
    - "Taylor Swift lyrics about..."
    - "songs that capture..."
    - "lyrics that express..."
    """
    if not isinstance(q, str):
        return ""

    q = q.strip()

    # Remove search-style openings like:
    # "Taylor Swift lyrics about..."
    # "songs that capture..."
    # "a lyric that expresses..."
    q = re.sub(
        r"^(taylor swift\s+)?(a\s+|the\s+)?"
        r"(song|songs|lyric|lyrics)\s+"
        r"(that\s+)?"
        r"(about|capture|captures|express|expresses|reflect|reflects|show|shows|assert|asserts|convey|conveys)\s+",
        "",
        q,
        flags=re.IGNORECASE
    )

    # Remove generic filler openings.
    q = re.sub(
        r"^(the\s+)?(feeling|emotion|experience|situation)\s+of\s+",
        "",
        q,
        flags=re.IGNORECASE
    )

    q = q.strip(" .,:;-")
    return q


def split_document(document):
    """
    Split the stored Chroma document into profile text and actual lyric text.

    Expected document format:

    Song: ...
    Album: ...
    Section: ...

    Song Profile:
    ...

    Lyrics:
    ...

    The reranker should score primarily from the Lyrics section.
    """
    if not isinstance(document, str):
        return "", ""

    marker = "Lyrics:"

    if marker not in document:
        return document.strip(), document.strip()

    profile_text, lyrics_text = document.split(marker, 1)

    return profile_text.strip(), lyrics_text.strip()


def meaningful_word_count(text):
    """
    Count meaningful words in a lyric section.

    This helps filter chunks like:
    'Oh-oh / Oh-oh'
    which are not useful final matches.
    """
    if not isinstance(text, str):
        return 0

    words = re.findall(r"[A-Za-z']+", text.lower())

    filler_words = {
        "oh", "ooh", "ah", "yeah", "hey", "ha", "la", "na",
        "mm", "mmm", "whoa", "woah"
    }

    meaningful_words = [
        word for word in words
        if word not in filler_words and len(word) > 1
    ]

    return len(meaningful_words)


def has_enough_lyric_evidence(document, min_words=8):
    """
    Return True if the actual lyric section has enough meaningful content
    to be judged as a final match.
    """
    _, lyrics_text = split_document(document)
    return meaningful_word_count(lyrics_text) >= min_words


# -----------------------------
# Search planning
# -----------------------------

def build_search_plan(user_query):
    """
    Create a general search plan.

    The key idea:
    - retrieve using natural situation/emotion queries
    - identify the user's target narrator state
    - later, rerank by whether the lyric narrator matches that state

    This avoids hardcoding specific cases like:
    "if user is over an ex, penalize longing."
    Instead, it works generally for romance, grief, friendship, confidence,
    anxiety, career stress, betrayal, nostalgia, etc.
    """
    prompt = f"""
User situation:
{user_query}

Create a concise search plan for finding lyric sections that match this situation.

Focus on:
- what is actually happening
- the user's emotional stance
- the user's POV
- the user's timeline/phase
- what the matching lyric narrator should feel, want, or need
- what kinds of lyric sections would feel validating
- what kinds of lyric sections would be misleading

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
    "emotional_state": "what the matching lyric narrator should feel",
    "agency_level": "passive | conflicted | setting_boundaries | taking_action | detached | reflective",
    "timeline_phase": "before_event | during_event | immediate_aftermath | unresolved | healing | moved_on | reflective_closure | uncertain",
    "speaker_role": "the role/perspective the lyric narrator should have"
  }},
  "good_match_signals": ["signal1", "signal2", "signal3"],
  "avoid_match_signals": ["signal1", "signal2", "signal3"],
  "queries": [
    "first-person natural situation query",
    "first-person emotional state query",
    "first-person agency or boundary query",
    "first-person timeline query"
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
        "good_match_signals": plan.get("good_match_signals", []),
        "avoid_match_signals": plan.get("avoid_match_signals", []),
        "queries": queries[:5]
    }


# -----------------------------
# Retrieval
# -----------------------------

def query_one(collection, query, fetch_k=35):
    """
    Run one Chroma query and normalize the response.
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


def retrieve_candidates(collection, search_plan, max_candidates=40, max_chunks_per_song=3):
    """
    Retrieve candidates from Chroma using the search plan queries.

    Retrieval should be generous. The reranker can reject weak matches, but it
    cannot rescue good matches that were filtered out too early.

    This also filters chunks with too little actual lyric evidence, so profile-only
    matches like outro filler do not reach the reranker.
    """
    raw_matches = []

    print("Search plan:")
    print(json.dumps(search_plan, indent=2))

    print("\nSearch queries:")
    for query in search_plan["queries"]:
        print(f"- {query}")
        raw_matches.extend(query_one(collection, query, fetch_k=35))

    raw_matches.sort(key=lambda m: m["distance"])

    filtered = []
    seen_docs = set()
    chunks_per_song = defaultdict(int)

    for match in raw_matches:
        doc = match["document"]
        song = match["metadata"].get("song", "")

        if not has_enough_lyric_evidence(doc, min_words=8):
            continue

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

    print(f"\nCandidates sent to reranker: {len(filtered)}")
    return filtered


# -----------------------------
# Reranking
# -----------------------------

def rerank_matches(user_query, search_plan, matches, final_k=5, min_score=7):
    """
    Rerank candidates by concrete situation fit, narrator state, timeline,
    speaker role, and lyric evidence.

    The most important check is not just:
    "Is this lyric about a similar topic?"

    It is:
    "Does this lyric speak from the same emotional position as the user?"
    """
    candidates = []

    for i, match in enumerate(matches):
        meta = match["metadata"]
        profile_text, lyrics_text = split_document(match["document"])

        candidates.append({
            "index": i,
            "song": meta.get("song", ""),
            "album": meta.get("album", ""),
            "section": meta.get("section", ""),
            "profile_text": profile_text,
            "lyrics_text": lyrics_text,
            "meaningful_lyric_word_count": meaningful_word_count(lyrics_text),
            "metadata": {
                "themes": meta.get("themes", ""),
                "moods": meta.get("moods", ""),
                "situations": meta.get("situations", ""),
                "perspective": meta.get("perspective", ""),
                "speaker_role": meta.get("speaker_role", ""),
                "narrator_agency": meta.get("narrator_agency", ""),
                "narrator_hurt_status": meta.get("narrator_hurt_status", ""),
                "relationship_stage": meta.get("relationship_stage", ""),
                "timeline_state": meta.get("timeline_state", ""),
                "summary": meta.get("summary", "")
            },
            "distance": match["distance"],
            "source_query": match["source_query"]
        })

    prompt = f"""
User situation:
{user_query}

Search plan:
{json.dumps(search_plan, indent=2)}

You are ranking lyric candidates.

Pick lyric sections that match the user's actual situation, POV, emotional stance, narrator state, and timeline.

Each candidate has:
- profile_text: song/profile metadata and summary
- lyrics_text: the actual lyric section being evaluated
- meaningful_lyric_word_count: count of non-filler lyric words

Score primarily from lyrics_text.
Use profile_text only as secondary context.
A strong profile_text cannot rescue weak lyrics_text.
If lyrics_text is vague, generic, repetitive filler, or too short to express the match, lower the score.

Evaluation rules:

1. Concrete situation
- What is actually happening to the user?
- Does lyrics_text match that situation, or only a similar emotion?
- Do not treat neighboring situations as the same.
  Examples:
  - being over someone vs still missing them
  - setting a boundary vs wanting reconciliation
  - public shame vs private heartbreak
  - grieving a death/loss vs missing an ex
  - revenge vs empowerment
  - anxiety/self-doubt vs romantic rejection
  - active crisis vs reflective closure

2. Narrator state alignment
- Compare the user's target narrator state against the lyric narrator's state.
- The lyric narrator should match the user's emotional state, agency level, timeline phase, and speaker role.
- Penalize wrong-state matches even when the general topic is similar.
- A candidate is a wrong-state match if the lyric narrator wants, feels, or needs something opposite from the user.
- Examples:
  - User is detached, but lyric narrator is desperate.
  - User is grieving, but lyric narrator is carefree.
  - User is anxious and uncertain, but lyric narrator is fully confident.
  - User is setting a boundary, but lyric narrator is seeking reconciliation.
  - User feels betrayed, but lyric narrator is the betrayer.
  - User wants escape, but lyric narrator wants to stay.
  - User is in the immediate crisis, but lyric narrator is calmly reflecting years later.
- Wrong-state matches should usually score 5 or below unless the lyric has unusually strong metaphorical fit.

3. Timeline / phase alignment
- Match where the user is emotionally in the situation.
- Do not treat the same event as equivalent across all phases.
- Before, during, immediate aftermath, unresolved pain, healing, moved-on detachment, and reflective closure are different phases.
- A lyric can share the same topic but still be wrong if it speaks from the wrong phase.

4. Lyric evidence
- First identify what lyrics_text is literally about.
- If lyrics_text is vague, generic, too short, or only loosely related, lower the score.
- Score lyrics_text, not the whole song.
- Do not let profile_text override weak lyric evidence.
- A famous song with a matching overall theme should not receive a high score unless lyrics_text itself contains strong evidence.

5. Score calibration
- Be strict, but practical.
- 10 is rare.
- Do not give high scores just because lyrics_text shares a mood word.
- Prefer 2-3 strong matches over filling the list with weak ones.

Scoring:
- 10: Exact concrete situation, same narrator state, same timeline phase, strong lyric evidence.
- 9: Very strong match with only tiny differences.
- 7-8: Good match; narrator state and timeline are right, even if the situation is somewhat metaphorical.
- 6: Usable near-match; tone or theme fits, but concrete situation is not exact.
- 5: General mood/theme match, but narrator state, role, or timeline is noticeably off.
- 1-4: Wrong narrator state, wrong speaker role, wrong timeline, mostly keyword overlap, or weak lyric evidence.

Important:
- Do not score a candidate 7+ if the narrator state conflicts with the user state.
- Do not score a candidate 7+ just because it contains related keywords.
- lyrics_text itself must support the score.
- Do not use your outside knowledge of the full song.
- Score only the provided lyrics_text.

Return JSON only.
Return one ranking entry for every candidate.

Format:
{{
  "rankings": [
    {{
      "index": 0,
      "score": 8,
      "match_type": "exact_situation_match | close_state_match | usable_near_match | general_theme_match | poor_match",
      "lyric_is_about": "one sentence describing what lyrics_text is literally about",
      "narrator_state": "what the lyric narrator seems to feel/want/need",
      "state_alignment": "same_state | close_state | partial_state | wrong_state",
      "analysis": "briefly explain situation fit, narrator state fit, timeline fit, and lyric evidence",
      "reason": "short user-facing explanation of why this lyric matches"
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
                "content": "You are a strict but practical lyric match judge. Return valid JSON only."
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
        match["lyric_is_about"] = item.get("lyric_is_about", "")
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
    Print the highest-scored candidates before final filtering.
    This makes debugging much easier when final output is unexpectedly weak
    or empty.
    """
    if not reranked:
        print("\nReranker returned no usable rankings.")
        return

    print("\nTop reranked candidates before final filtering:")
    for match in reranked[:top_n]:
        meta = match["metadata"]
        _, lyrics_text = split_document(match.get("document", ""))

        print(
            f"- {match.get('rerank_score', 0)}/10 | "
            f"{meta.get('song', '')} | {meta.get('section', '')} | "
            f"{match.get('match_type', '')} | "
            f"{match.get('state_alignment', '')} | "
            f"words={meaningful_word_count(lyrics_text)}"
        )


def select_final_matches(reranked, final_k=5, min_score=7):
    """
    Select confident, diverse matches.

    Main behavior:
    - require min_score
    - avoid duplicate songs
    - avoid near-duplicate lyric sections
    - avoid wrong-state matches
    - avoid lyric sections with too little meaningful evidence

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

        if not has_enough_lyric_evidence(match.get("document", ""), min_words=8):
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

    # Fallback: keep the best diverse candidates with scores >= 6.
    # These should be displayed as weaker matches, not as perfect recommendations
    fallback = []
    seen_songs = set()

    for match in reranked:
        if match.get("rerank_score", 0) < 6:
            continue

        if match.get("state_alignment") == "wrong_state":
            continue

        if not has_enough_lyric_evidence(match.get("document", ""), min_words=8):
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
    for match in matches:
        meta = match["metadata"]
        profile_text, lyrics_text = split_document(match.get("document", ""))
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
        print(f"Lyric is about: {match.get('lyric_is_about', '')}")
        print(f"Narrator State: {match.get('narrator_state', '')}")
        print(f"State Alignment: {match.get('state_alignment', '')}")
        print(f"Meaningful Lyric Words: {meaningful_word_count(lyrics_text)}")
        print(f"Analysis: {match.get('analysis', '')}")
        print(f"Why: {match.get('reason', '')}")
        print()
        print("Lyrics:")
        print(lyrics_text)
        print()


def run_lyric_search(user_query, num_results=5, min_score=7):
    """
    Main entry point for CLI and Streamlit.
    """
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    search_plan = build_search_plan(user_query)

    candidates = retrieve_candidates(
        collection=collection,
        search_plan=search_plan,
        max_candidates=25,
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
    query = "I wanna move to a new city alone but I’m anxious I’ll be lonely"

    matches = run_lyric_search(
        user_query=query,
        num_results=5,
        min_score=7
    )

    print(f"\nFinal matches: {len(matches)}\n")
    print_matches(matches)

    save_search_log(
        user_query=query,
        matches=matches,
        jsonl_filename="search_logs2.jsonl",
        pretty_filename="search_logs_pretty2.txt"
    )

    print("Saved logs to search_logs2.jsonl and search_logs_pretty2.txt")


if __name__ == "__main__":
    main()