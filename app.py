import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TuneMatch 🎵",
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

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(138, 43, 226, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(255, 20, 147, 0.15),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #090014,
            #12001f 45%,
            #08000f
        );

    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* HERO */

.hero {
    text-align: center;
    padding: 40px 20px 30px;
}

.hero-icon {
    font-size: 65px;
    margin-bottom: 5px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
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

.hero p {
    font-size: 18px;
    color: #b9aec7;
    margin-top: 10px;
}


/* SEARCH BOX */

.search-box {
    background: rgba(255,255,255,0.06);

    border: 1px solid rgba(255,255,255,0.12);

    border-radius: 22px;

    padding: 25px;

    margin: 20px auto 25px;

    max-width: 900px;

    box-shadow:
        0 10px 40px rgba(0,0,0,0.25);
}


/* BUTTON */

.stButton > button {

    background: linear-gradient(
        90deg,
        #a855f7,
        #ec4899
    );

    color: white;

    border: none;

    border-radius: 12px;

    font-weight: 600;

    padding: 12px;

    transition: 0.3s;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(168,85,247,0.4);
}


/* RECOMMENDATION CARD */

.card {

    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.08),
        rgba(255,255,255,0.025)
    );

    border: 1px solid rgba(255,255,255,0.1);

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 15px;

    transition: all 0.25s ease;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.2);
}

.card:hover {

    transform: translateY(-4px);

    border-color: rgba(168,85,247,0.6);

    box-shadow:
        0 12px 35px rgba(168,85,247,0.18);
}

.song-number {

    font-size: 14px;

    color: #a855f7;

    font-weight: 600;
}

.song-title {

    font-size: 20px;

    font-weight: 600;

    color: white;
}

.artist {

    font-size: 14px;

    color: #aaa0b5;

    margin-top: 4px;
}

.music-icon {

    font-size: 30px;
}


/* SECTION */

.section-title {

    font-size: 28px;

    font-weight: 700;

    margin-top: 35px;

    margin-bottom: 5px;
}


/* FOOTER */

.footer {

    text-align: center;

    color: #776d82;

    font-size: 13px;

    margin-top: 50px;
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


df, tfidf_matrix = load_data()


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_songs(song_name, top_n=5):

    # Find exact song
    matches = df[
        df["song"].str.lower() == song_name.lower()
    ]

    if len(matches) == 0:
        return None

    # Get index
    idx = matches.index[0]

    # Calculate similarity only for selected song
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Sort highest similarity first
    similar_indices = similarity_scores.argsort()[::-1]

    # Remove selected song
    similar_indices = similar_indices[
        similar_indices != idx
    ]

    # Select top N
    similar_indices = similar_indices[:top_n]

    recommendations = df[
        ["artist", "song"]
    ].iloc[similar_indices].copy()

    return recommendations.reset_index(drop=True)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-icon">🎧</div>

    <h1>TuneMatch</h1>

    <p>
        Find your next favorite song
        <br>
        <span style="color:#a855f7;">
            Powered by Machine Learning
        </span>
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SEARCH SECTION
# ============================================================

st.markdown("""
<div class="search-box">

    <h3 style="text-align:center; font-size:24px;">
        🎵 What are you listening to?
    </h3>

    <p style="
        text-align:center;
        color:#aaa0b5;
        font-size:14px;
        margin-bottom:0;
    ">
        Choose a song and we'll find tracks with a similar vibe.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SONG SELECTION
# ============================================================

song_list = sorted(
    df["song"]
    .dropna()
    .unique()
)


selected_song = st.selectbox(
    "Choose a song",
    song_list,
    index=None,
    placeholder="🔎 Search for a song..."
)


# ============================================================
# RECOMMEND BUTTON
# ============================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    recommend_button = st.button(
        "✨ Recommend Songs",
        use_container_width=True
    )


# ============================================================
# RESULTS
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
                "❌ Song not found in the dataset."
            )

        else:

            st.markdown(
    f"""
    <div class="section-title">
        ✨ Your Recommendations
    </div>

    <p style="color:#aaa0b5;">
        Songs similar to
        <b style="color:white;">
            {selected_song}
        </b>
    </p>
    """,
    unsafe_allow_html=True
)
                <p style="color:#aaa0b5;">

                    Based on

                    <b style="color:white;">
                        {selected_song}
                    </b>

                </p>
                """,
                unsafe_allow_html=True
            )


            # ==================================================
            # RECOMMENDATION CARDS
            # ==================================================

            for i, row in recommendations.iterrows():

                st.markdown(
                    f"""
                    <div class="card">

                        <div style="
                            display:flex;
                            align-items:center;
                            gap:18px;
                        ">

                            <div class="music-icon">
                                🎵
                            </div>

                            <div style="flex:1;">

                                <div class="song-number">
                                    #{i + 1}
                                </div>

                                <div class="song-title">
                                    {row['song']}
                                </div>

                                <div class="artist">
                                    🎤 {row['artist']}
                                </div>

                            </div>

                            <div style="
                                font-size:25px;
                            ">
                                ▶️
                            </div>

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

    🎧 <b>TuneMatch</b>

    <br>

    <span style="color:#665d70;">
        Music Recommendation System
    </span>

    <br><br>

    Built with Python • Pandas • Scikit-learn • Streamlit

</div>
""", unsafe_allow_html=True))
