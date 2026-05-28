# app.py
import streamlit as st
from retrieve_and_rerank import run_lyric_search

st.set_page_config(
    page_title="Taylor Swift Lyric Match",
    page_icon="🎵",
    layout="centered"
)


def extract_lyrics(document):
    """
    Your embedded document includes profile text + lyrics.
    This pulls out only the lyric section for display.
    """
    marker = "Lyrics:"
    if marker in document:
        return document.split(marker, 1)[1].strip()
    return document.strip()


st.title("Taylor Swift Lyric Match")
st.caption("Describe a situation. Get lyric matches ranked by emotion, POV, and timeline.")

user_input = st.text_area(
    "What are you going through?",
    placeholder="Example: I just got dumped and it came out of nowhere.",
    height=100
)

num_results = st.slider("Number of matches", min_value=3, max_value=10, value=5)

search_clicked = st.button("Find lyrics", type="primary")

if search_clicked:
    if not user_input.strip():
        st.warning("Enter a situation first.")
    else:
        with st.spinner("Finding the best lyrical matches..."):
            results = run_lyric_search(user_input, num_results=num_results)

        if not results:
            st.error("No matches found. Try describing the situation another way.")
        else:
            for i, match in enumerate(results, start=1):
                meta = match["metadata"]
                lyrics = extract_lyrics(match["document"])

                score = match.get("rerank_score", "N/A")
                distance = match.get("distance", 0)

                with st.container(border=True):
                    st.markdown(
                        f"### {i}. {meta.get('song', '')}"
                    )

                    st.caption(
                        f"{meta.get('album', '')} • {meta.get('section', '')} • "
                        f"Score: {score}/10 • Distance: {distance:.4f}"
                    )

                    st.markdown(f"> {lyrics.replace(chr(10), '<br>')}", unsafe_allow_html=True)

                    st.info(match.get("reason", ""))

                    with st.expander("Why this matched"):
                        st.write("**Match type:**", match.get("match_type", ""))
                        st.write("**Situation analysis:**", match.get("situation_analysis", ""))
                        st.write("**POV / timeline analysis:**", match.get("pov_timeline_analysis", ""))

                        st.write("**Speaker role:**", meta.get("speaker_role", ""))
                        st.write("**Agency:**", meta.get("narrator_agency", ""))
                        st.write("**Hurt status:**", meta.get("narrator_hurt_status", ""))
                        st.write("**Timeline:**", meta.get("timeline_state", ""))