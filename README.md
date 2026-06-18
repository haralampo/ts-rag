# Taylor Swift RAG Lyric Match

A RAG-style semantic search project that matches user-described emotions or situations to relevant lyric sections from a custom dataset.

The goal of this project is not just to search lyrics by keyword. It is to retrieve lyric sections that match the user's emotional state, point of view, timeline, and narrative situation.
Retrieval uses a two-stage pipeline: a vector similarity search over ChromaDB (embedded with all-MiniLM-L6-v2) surfaces candidate lyric sections, and an LLM reranker (GPT-4o-mini) scores each candidate across narrator state, speaker role, timeline phase, and lyric evidence. This second stage is what allows the system to distinguish between situations that are topically similar but narratively opposite.

> Read the full build/debugging write-up: [Building a Taylor Swift RAG Program](https://open.substack.com/pub/haralampo/p/sad-beautiful-tragic?r=7yj7sg&utm_campaign=post&utm_medium=web)

---

## Live Demo

[Try the public demo here](https://ts-rag-public-demo.streamlit.app/#2-question)

Note: the public demo hides lyric text for copyright reasons. Thus, those matches are based on song-section metadata rather than raw lyrics; some results may be less precise. 

The first search may take up to a minute while the app loads and runs retrieval/reranking.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Tech Stack](#tech-stack)
* [How It Works](#how-it-works)
* [Technical Challenges](#technical-challenges)
* [Project Architecture](#project-architecture)
* [Repository Structure](#repository-structure)
* [Setup](#setup)
* [Running the Pipeline](#running-the-pipeline)
* [Example Query](#example-query)
* [What I Learned](#what-i-learned)
* [Future Improvements](#future-improvements)
* [Disclaimer](#disclaimer)

---

## Project Overview

Taylor Swift RAG Lyric Match is a semantic retrieval system for matching real-life situations to Taylor Swift lyric sections.

A user can enter something like:

```text
my ex won't move on even though I'm over him
```

A basic semantic search system might retrieve breakup lyrics about grief, longing, or not being able to move on. This project tries to distinguish between similar-sounding but narratively different situations:

```text
"I can't move on from my ex"
```

versus:

```text
"My ex can't move on from me, and I want distance"
```

Those scenarios use similar words, but they require different lyric matches.

---

## Tech Stack

* Python
* Pandas
* ChromaDB
* SentenceTransformers
* `all-MiniLM-L6-v2`
* OpenAI API
* `gpt-4o-mini`
* python-dotenv
* JSON / CSV
* Optional UI: Streamlit

---

## How It Works

### 1. Clean and Chunk Lyrics

Raw lyric files are cleaned and split into smaller sections using headers like:

```text
[Verse 1]
[Chorus]
[Bridge]
[Outro]
```

Each chunk stores:

```text
song
album
section
text
```

Chunking by section improves retrieval accuracy because a full song can contain multiple emotional states. 

### 2. Generate Song Profiles for the Lyric Dataset

The pipeline generates song-level profiles that summarize each song’s broader emotional and narrative context.

Profiles may include information like:

```text
themes
moods
situations
perspective
speaker role
timeline state
summary
```

These profiles help the system understand context that may not be obvious from a short lyric section alone.

### 3. Combine Profiles With Lyric Chunks for Embedding

During ingestion, the song profile and the specific lyric chunk are combined into one document before being embedded and stored in ChromaDB.

The embedded document contains both:

```text
song-level profile context
+
actual lyric section
```

This gives vector search more context than the lyric chunk alone. During reranking, the system separates the profile text from the lyric text so the final score is based primarily on the retrieved lyric section, not the song-level metadata.

### 4. Embed and Store in ChromaDB

Each combined document is converted into a vector using:

```text
all-MiniLM-L6-v2
```

The embeddings and metadata are stored in a local ChromaDB collection.

When the user enters a scenario, the system embeds the query and retrieves candidate documents that are semantically similar.

### 5. Build a Search Plan for the User Query

Before retrieval, an LLM creates a lightweight search plan from the user’s input.

The search plan identifies:

* what is happening
* the user’s emotional stance
* the user’s point of view
* the timeline or emotional phase
* good match signals
* avoid match signals
* natural retrieval queries

This helps prevent the system from relying only on surface-level similarity.

### 6. Retrieve Candidate Matches

The search plan generates several natural-language retrieval queries.

ChromaDB uses those queries to retrieve candidate lyric sections from the vector database.

The retrieval step also filters out obvious weak candidates, such as duplicate chunks or sections with too little meaningful lyric evidence (ex. "Uh-huh").

### 7. Rerank by State and Lyric Evidence

After retrieval, an LLM reranks the candidate lyric sections.

The stored document is split into:

```text
profile_text
lyrics_text
```

The profile provides secondary context, but the final score is based primarily on the actual lyric section.

The reranker evaluates:

* concrete situation match
* emotional alignment
* narrator point of view
* timeline alignment
* speaker role
* agency
* actual lyric evidence

This helps filter out lyrics that are generally related but narratively wrong.

### 8. Return Confident Matches or Near-Matches

The system prefers fewer strong matches over a full list of weak ones.

If there are not enough high-confidence results, it returns clearly labeled low-confidence near-matches instead of silently failing or filling the output with misleading recommendations.

### 9. Log Searches

Searches can be logged to:

```text
search_logs.jsonl
search_logs_pretty.txt
```

This makes it easier to compare outputs across versions and evaluate how changes affected retrieval quality.

---

## Technical Challenges

### Semantic Similarity Was Not Enough

Early retrieval matched broad emotional themes, but missed subtle narrator-state differences (ex. "I broke up with my ex" vs. "my ex broke up with me").

For example:

```text
my ex won't move on even though I'm over him
```

The fix was to rerank by point of view, timeline, speaker role, and emotional stance.

### Query Expansion Introduced Drift

Query expansion helped retrieval, but some generated queries became too generic:

```text
Taylor Swift lyrics about moving on
Taylor Swift lyrics expressing annoyance
Taylor Swift lyrics about being done with someone
```

These were weak because the vector database contains lyric chunks, not web pages. 

The fix was to have the queries use more natural language:

```text
I am done explaining myself and want emotional distance.
```

### Song Profiles Could Overpower Lyric Evidence

Song-level profiles helped add context, but they sometimes caused weak chunks to rank too highly.

For example, a short filler chunk like:

```text
Mmm-hmm
```

could rank because the song profile matched the query, even though the lyric itself had no useful evidence.

The solution was to split the retrieved document back into profile text and lyric text during reranking, then score primarily from the lyric section itself.

### Strict Filtering Created Empty Results

A stricter reranker reduced bad matches, but sometimes returned too few results.

The fix was to support clearly labeled low-confidence near-matches instead of silently returning nothing or filling the UI with weak results.

---

## Project Architecture

```text
Raw lyric files
      ↓
clean_lyrics.py
      ↓
Chunked CSV files
      ↓
generate_song_profiles.py
      ↓
ingest_lyrics.py
      ↓
Profile + lyric chunk documents
      ↓
ChromaDB vector database
      ↓
retrieve_and_rerank.py
      ↓
State-aware ranked lyric matches
      ↓
Search logs / optional UI
```

---

## Repository Structure

```text
taylor-swift-rag/
│
├── data/
│   ├── raw/
│   │   ├── evermore.txt
│   │   ├── folklore.txt
│   │   └── ...
│   │
│   ├── chunks/
│   │   ├── evermore_chunks.csv
│   │   ├── folklore_chunks.csv
│   │   └── ...
│   │
│   └── song_profiles.json
│
├── logs/
│   ├── search_logs.jsonl
│   └── search_logs_pretty.txt
│
├── clean_lyrics.py
├── generate_song_profiles.py
├── ingest_lyrics.py
├── retrieve_and_rerank.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/taylor-swift-rag.git
cd taylor-swift-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```bash
touch .env
```

Add your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Pipeline

### 1. Clean and chunk lyrics

```bash
python clean_lyrics.py
```

### 2. Generate song profiles

```bash
python generate_song_profiles.py
```

### 3. Ingest chunks into ChromaDB

```bash
python ingest_lyrics.py
```

### 4. Retrieve and rerank matches

```bash
python retrieve_and_rerank.py
```

---

## Example Query

```text
my ex won't move on even though I'm over him
```

The system should prefer matches where the narrator is:

* detached
* done
* setting boundaries
* reacting to unwanted emotional attachment

The system should avoid matches where the narrator is:

* still heartbroken
* begging for someone to return
* unable to move on
* hoping for reconciliation

---

## What I Learned

This project taught me that RAG quality is not just about storing documents and retrieving similar text.

The harder problem was distinguishing between similar topic and same situation. A strong match needs to align with the user’s point of view, timeline, emotional stance, speaker role, and actual situation.

The final version is more reliable because it combines vector search with state-aware reranking and search logging.

---

## Future Improvements

* Build a Streamlit interface
* Add a game mode where users guess the best song for a generated scenario
* Add user ratings to evaluate match quality
* Create a benchmark set of test scenarios
* Add filters by album, era, or emotional category
* Deploy the app

---

## Data Notice

This repository does not include copyrighted lyric files, generated lyric chunks, search logs containing lyrics, or the local ChromaDB database. To run the pipeline, provide your own local dataset or replace the input files with placeholder/sample text.

---

## Disclaimer

This project is for educational and portfolio purposes only. It is not affiliated with Taylor Swift, TAS Rights Management, Republic Records, or any related entities. 