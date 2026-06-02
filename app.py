# app.py
import re

import streamlit as st
from retrieve_and_rerank import run_lyric_search

st.set_page_config(
    page_title="Taylor Swift Lyric Match",
    page_icon="🎵",
    layout="centered"
)


def extract_lyrics(document):
    """
    The embedded document includes profile text + lyrics.
    This pulls out only the lyric section for display.
    """
    marker = "Lyrics:"
    if marker in document:
        return document.split(marker, 1)[1].strip()
    return document.strip()


def meaningful_word_count(text):
    """
    Count meaningful lyric words while ignoring filler like oh/ooh/yeah.
    Useful for debugging weak lyric chunks.
    """
    words = re.findall(r"[A-Za-z']+", text.lower())

    filler_words = {
        "oh", "ooh", "ah", "yeah", "hey", "ha", "la", "na",
        "mm", "mmm", "whoa", "woah"
    }

    return len([
        word for word in words
        if word not in filler_words and len(word) > 1
    ])


st.title("Taylor Swift Lyric Match")
st.caption("Describe a situation. Get lyric matches ranked by emotion, POV, and timeline.")

user_input = st.text_area(
    "What are you going through?",
    placeholder="Example: I just got dumped and it came out of nowhere.",
    height=100
)

num_results = st.slider(
    "Number of matches",
    min_value=3,
    max_value=10,
    value=5
)

search_clicked = st.button("Find lyrics", type="primary")

if search_clicked:
    if not user_input.strip():
        st.warning("Enter a situation first.")
    else:
        with st.spinner("Finding the best lyrical matches..."):
            results = run_lyric_search(
                user_query=user_input,
                num_results=num_results
            )

        if not results:
            st.error("No matches found. Try describing the situation another way.")
        else:
            for i, match in enumerate(results, start=1):
                meta = match.get("metadata", {})
                lyrics = extract_lyrics(match.get("document", ""))

                score = match.get("rerank_score", "N/A")
                distance = match.get("distance", 0)

                with st.container(border=True):
                    st.markdown(f"### {i}. {meta.get('song', '')}")

                    st.caption(
                        f"{meta.get('album', '')} • "
                        f"{meta.get('section', '')} • "
                        f"Score: {score}/10 • "
                        f"Distance: {distance:.4f} • "
                        f"Words: {meaningful_word_count(lyrics)}"
                    )

                    st.markdown(
                        f"> {lyrics.replace(chr(10), '<br>')}",
                        unsafe_allow_html=True
                    )

                    reason = match.get("reason", "")
                    if reason:
                        st.info(reason)

                    with st.expander("Why this matched"):
                        st.write("**Match type:**", match.get("match_type", ""))
                        st.write("**Lyric is about:**", match.get("lyric_is_about", ""))
                        st.write("**Narrator state:**", match.get("narrator_state", ""))
                        st.write("**State alignment:**", match.get("state_alignment", ""))
                        st.write("**Analysis:**", match.get("analysis", ""))

                        st.divider()

                        st.write("**Speaker role:**", meta.get("speaker_role", ""))
                        st.write("**Agency:**", meta.get("narrator_agency", ""))
                        st.write("**Hurt status:**", meta.get("narrator_hurt_status", ""))
                        st.write("**Timeline:**", meta.get("timeline_state", ""))