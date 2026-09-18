
import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import html


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TuneMatch",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(168, 85, 247, 0.22),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 25%,
            rgba(236, 72, 153, 0.18),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #07000d 0%,
            #12001f 50%,
            #050008 100%
        );

    color: white;
}


/* MAIN CONTAINER */

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-container {
    text-align: center;
    padding: 45px 20px 35px;
}

.hero-icon {
    font-size: 70px;
    margin-bottom: 5px;
    filter: drop-shadow(
        0 0 20px rgba(168, 85, 247, 0.7)
    );
}

.hero-title {
    font-size: 56px;
    font-weight: 800;
    letter-spacing: -2px;
    margin: 0;

    background: linear-gradient(
        90deg,
        #ff4ecd,
        #a855f7,
        #6366f1
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 18px;
    color: #b7aabd;
    margin-top: 12px;
}

.hero-highlight {
    color: #c084fc;
    font-weight: 600;
}


/* ============================================================
   SEARCH PANEL
   ============================================================ */

.search-panel {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.12);

    border-radius: 24px;

    padding: 30px;

    margin: 10px auto 25px;

    box-shadow:
        0 15px 50px rgba(0, 0, 0, 0.35);

    backdrop-filter: blur(12px);
}

.search-title {
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 8px;
}

.search-description {
    text-align: center;
    color: #aaa0b5;
    font-size: 14px;
    margin-bottom: 20px;
}


/* ============================================================
   STREAMLIT SELECTBOX
   ============================================================ */

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
}

div[data-baseweb="select"] span {
    color: white !important;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {

    width: 100%;

    background: linear-gradient(
        90deg,
        #9333ea,
        #ec4899
    );

    color: white;

    border: none;

    border-radius: 13px;

    padding: 13px 20px;

    font-size: 16px;

    font-weight: 600;

    transition: all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow:
        0 10px 30px rgba(168, 85, 247, 0.45);
}


/* ============================================================
   RESULT HEADER
   ============================================================ */

.results-header {
    margin-top: 35px;
    margin-bottom: 20px;
}

.results-title {
    font-size: 30px;
    font-weight: 700;
}

.results-description {
    color: #aaa0b5;
    font-size: 14px;
}


/* ============================================================
   SONG CARD
   ============================================================ */

.song-card {

    display: flex;

    align-items: center;

    gap: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.09),
            rgba(255,255,255,0.025)
        );

    border: 1px solid rgba(255,255,255,0.10);

    border-radius: 18px;

    padding: 18px 20px;

    margin-bottom: 14px;

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

.song-card:hover {

    transform: translateY(-4px);

    border-color: rgba(168,85,247,0.65);

    box-shadow:
        0 12px 35px rgba(168,85,247,0.18);
}

.song-icon {

    width: 55px;
    height: 55px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            rgba(168,85,247,0.25),
            rgba(236,72,153,0.20)
        );

    font-size: 26px;
}

.song-info {
    flex: 1;
}

.song-rank {
    color: #c084fc;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 3px;
}

.song-name {
    color: white;
    font-size: 19px;
    font-weight: 600;
}

.song-artist {
    color: #9f94a8;
    font-size: 13px;
    margin-top: 3px;
}

.play-icon {
    font-size: 23px;
    opacity: 0.85;
}


/* ============================================================
   EMPTY / INFO BOX
   ============================================================ */

.info-box {

    text-align: center;

    padding: 30px;

    border-radius: 18px;

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.08);

    color: #aaa0b5;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    text-align: center;

    margin-top: 55px;

    padding-top: 25px;

    border-top: 1px solid rgba(255,255,255,0.08);

    color: #756b80;

    font-size: 13px;
}

.footer-title {
    color: #b9a8c8;
    font-size: 15px;
    font-weight: 600;
}

.footer-highlight {
    color: #a855f7;
}


</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    df = pd.read_csv("songs.csv")

    tfidf_matrix = load_npz("tfidf_matrix.npz")

    return df, tfidf_matrix


# ============================================================
# LOAD DATA SAFELY
# ============================================================

try:

    df, tfidf_matrix = load_data()

except FileNotFoundError:

    st.error(
        "❌ Required files were not found. "
        "Make sure songs.csv and tfidf_matrix.npz "
        "are in the same folder as app.py."
    )

    st.stop()


# ============================================================
# CHECK DATA
# ============================================================

required_columns = {"artist", "song"}

if not required_columns.issubset(df.columns):

    st.error(
        "❌ songs.csv must contain these columns: "
        "artist and song"
    )

    st.stop()


if len(df) != tfidf_matrix.shape[0]:

    st.error(
        "❌ The number of songs in songs.csv does not "
        "match the number of rows in tfidf_matrix.npz."
    )

    st.stop()


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_songs(song_name, top_n=5):

    song_values = (
        df["song"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    matches = df[
        song_values == song_name.strip().lower()
    ]

    if matches.empty:
        return None

    # Original dataframe index
    idx = matches.index[0]

    # Position inside TF-IDF matrix
    matrix_position = df.index.get_loc(idx)

    # Calculate similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[matrix_position],
        tfidf_matrix
    ).flatten()

    # Sort from highest similarity
    similar_positions = similarity_scores.argsort()[::-1]

    # Remove selected song
    similar_positions = [
        position
        for position in similar_positions
        if position != matrix_position
    ]

    # Take top N
    similar_positions = similar_positions[:top_n]

    recommendations = df.iloc[
        similar_positions
    ][["artist", "song"]].copy()

    return recommendations.reset_index(drop=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-container">

    <div class="hero-icon">
        🎧
    </div>

    <div class="hero-title">
        TuneMatch
    </div>

    <div class="hero-subtitle">
        Discover music that matches your vibe.
        <br>
        <span class="hero-highlight">
            Powered by Machine Learning
        </span>
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SEARCH PANEL
# ============================================================

st.markdown("""
<div class="search-panel">

    <div class="search-title">
        🎵 Find Your Next Favorite Song
    </div>

    <div class="search-description">
        Select a song below and TuneMatch will recommend
        tracks with a similar musical vibe.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SONG LIST
# ============================================================

song_list = sorted(
    df["song"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


selected_song = st.selectbox(
    "Select a song",
    song_list,
    index=None,
    placeholder="🔎 Search or select a song..."
)


# ============================================================
# RECOMMEND BUTTON
# ============================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    recommend_button = st.button(
        "✨ Find Similar Songs",
        use_container_width=True
    )


# ============================================================
# RESULTS
# ============================================================

if recommend_button:

    if selected_song is None:

        st.warning(
            "🎵 Please select a song before getting recommendations."
        )

    else:

        recommendations = recommend_songs(
            selected_song,
            top_n=5
        )

        if recommendations is None:

            st.error(
                "❌ The selected song could not be found."
            )

        elif recommendations.empty:

            st.info(
                "No similar songs were found."
            )

        else:

            safe_song = html.escape(
                str(selected_song)
            )

            # ------------------------------------------------
            # RESULTS HEADER
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="results-header">

                    <div class="results-title">
                        ✨ Recommended For You
                    </div>

                    <div class="results-description">
                        Similar songs based on
                        <b style="color:white;">
                            {safe_song}
                        </b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # SONG CARDS
            # ------------------------------------------------

            for i, row in recommendations.iterrows():

                safe_song_name = html.escape(
                    str(row["song"])
                )

                safe_artist = html.escape(
                    str(row["artist"])
                )

                st.markdown(
                    f"""
                    <div class="song-card">

                        <div class="song-icon">
                            🎵
                        </div>

                        <div class="song-info">

                            <div class="song-rank">
                                RECOMMENDATION #{i + 1}
                            </div>

                            <div class="song-name">
                                {safe_song_name}
                            </div>

                            <div class="song-artist">
                                🎤 {safe_artist}
                            </div>

                        </div>

                        <div class="play-icon">
                            ▶️
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <div class="footer-title">
        🎧 TuneMatch
    </div>

    <div>
        Music Recommendation System
    </div>

    <br>

    <div>
        Built with
        <span class="footer-highlight">Python</span>
        •
        <span class="footer-highlight">Pandas</span>
        •
        <span class="footer-highlight">Scikit-learn</span>
        •
        <span class="footer-highlight">Streamlit</span>
    </div>

</div>
""", unsafe_allow_html=True)

