
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
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap'
);

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(168, 85, 247, 0.20),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(236, 72, 153, 0.16),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #08000f,
            #12001f,
            #06000b
        );

    color: white;
}

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* ================= HERO ================= */

.hero {
    text-align: center;
    padding: 45px 20px 35px;
}

.hero-icon {
    font-size: 65px;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 55px;
    font-weight: 800;
    line-height: 1.1;

    background: linear-gradient(
        90deg,
        #f472b6,
        #a855f7,
        #6366f1
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #b8aabe;
    font-size: 17px;
    margin-top: 12px;
    line-height: 1.7;
}

.hero-highlight {
    color: #c084fc;
    font-weight: 600;
}


/* ================= SEARCH ================= */

.search-container {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;

    padding: 28px;
    margin-bottom: 25px;

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.30);

    backdrop-filter: blur(10px);
}

.search-title {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
}

.search-description {
    text-align: center;
    color: #aaa0b5;
    font-size: 14px;
    margin-top: 6px;
}


/* ================= SELECT BOX ================= */

div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

div[data-baseweb="select"] span {
    color: white !important;
}


/* ================= BUTTON ================= */

.stButton > button {
    background: linear-gradient(
        90deg,
        #9333ea,
        #ec4899
    );

    color: white;
    border: none;
    border-radius: 12px;

    padding: 12px;
    font-size: 16px;
    font-weight: 600;

    transition: 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(168, 85, 247, 0.45);
}


/* ================= RESULTS ================= */

.results-heading {
    margin-top: 35px;
    margin-bottom: 20px;
}

.results-title {
    font-size: 29px;
    font-weight: 700;
}

.results-subtitle {
    color: #aaa0b5;
    font-size: 14px;
    margin-top: 4px;
}


/* ================= SONG CARD ================= */

.song-card {
    display: flex;
    align-items: center;
    gap: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(255, 255, 255, 0.085),
            rgba(255, 255, 255, 0.025)
        );

    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;

    padding: 18px;
    margin-bottom: 14px;

    transition: 0.25s;
}

.song-card:hover {
    transform: translateY(-3px);

    border-color: rgba(168, 85, 247, 0.60);

    box-shadow:
        0 12px 30px rgba(168, 85, 247, 0.16);
}

.song-icon {
    width: 55px;
    height: 55px;
    min-width: 55px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 15px;

    background: rgba(168, 85, 247, 0.15);

    font-size: 25px;
}

.song-details {
    flex: 1;
}

.song-rank {
    color: #c084fc;
    font-size: 11px;
    font-weight: 600;
}

.song-name {
    color: white;
    font-size: 18px;
    font-weight: 600;
    margin-top: 2px;
}

.song-artist {
    color: #9f94a8;
    font-size: 13px;
    margin-top: 3px;
}

.play-icon {
    font-size: 22px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    margin-top: 55px;
    padding-top: 25px;

    border-top: 1px solid rgba(255, 255, 255, 0.08);

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


try:

    df, tfidf_matrix = load_data()

except FileNotFoundError:

    st.error(
        "❌ songs.csv or tfidf_matrix.npz was not found."
    )

    st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

if "song" not in df.columns or "artist" not in df.columns:

    st.error(
        "❌ songs.csv must contain 'song' and 'artist' columns."
    )

    st.stop()


if len(df) != tfidf_matrix.shape[0]:

    st.error(
        "❌ songs.csv and tfidf_matrix.npz have different "
        "numbers of rows."
    )

    st.stop()


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_songs(song_name, top_n=5):

    song_column = (
        df["song"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    matches = df[
        song_column == song_name.strip().lower()
    ]

    if matches.empty:
        return None

    original_index = matches.index[0]

    matrix_index = df.index.get_loc(
        original_index
    )

    similarity_scores = cosine_similarity(
        tfidf_matrix[matrix_index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = [
        i
        for i in similar_indices
        if i != matrix_index
    ]

    similar_indices = similar_indices[:top_n]

    recommendations = df.iloc[
        similar_indices
    ][["artist", "song"]].copy()

    return recommendations.reset_index(drop=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">🎧</div>

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
# SEARCH BOX
# ============================================================

st.markdown("""
<div class="search-container">

    <div class="search-title">
        🎵 Find Your Next Favorite Song
    </div>

    <div class="search-description">
        Select a song below and TuneMatch will find
        tracks with a similar musical vibe.
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SONG SELECTION
# ============================================================

song_list = sorted(
    df["song"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

selected_song = st.selectbox(
    "Choose a song",
    song_list,
    index=None,
    placeholder="🔎 Search for a song..."
)


# ============================================================
# BUTTON
# ============================================================

left, middle, right = st.columns([1, 2, 1])

with middle:

    recommend_button = st.button(
        "✨ Recommend Songs",
        use_container_width=True
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

if recommend_button:

    if selected_song is None:

        st.warning(
            "🎵 Please select a song first."
        )

    else:

        recommendations = recommend_songs(
            selected_song,
            top_n=5
        )

        if recommendations is None:

            st.error(
                "❌ Song not found."
            )

        else:

            safe_selected_song = html.escape(
                str(selected_song)
            )

            # Results heading
            st.markdown(
                f"""
                <div class="results-heading">

                    <div class="results-title">
                        ✨ Your Recommendations
                    </div>

                    <div class="results-subtitle">
                        Similar songs to
                        <b style="color:white;">
                            {safe_selected_song}
                        </b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # Song cards
            for number, row in recommendations.iterrows():

                safe_song = html.escape(
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

                        <div class="song-details">

                            <div class="song-rank">
                                RECOMMENDATION #{number + 1}
                            </div>

                            <div class="song-name">
                                {safe_song}
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

