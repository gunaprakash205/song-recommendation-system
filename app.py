
import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


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
# CUSTOM CSS - ONLY STYLING
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
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-box {
    text-align: center;
    padding: 40px 20px;
}

.hero-title {
    font-size: 55px;
    font-weight: 800;
    margin: 5px 0;
}

.hero-subtitle {
    color: #b8aabe;
    font-size: 17px;
}

.search-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 25px;
    margin: 20px 0;
}

.result-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 18px;
    margin: 10px 0;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding: 25px;
    color: #81768b;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    songs = pd.read_csv("songs.csv")

    matrix = load_npz("tfidf_matrix.npz")

    return songs, matrix


try:

    df, tfidf_matrix = load_data()

except FileNotFoundError:

    st.error(
        "❌ Could not find songs.csv or tfidf_matrix.npz."
    )

    st.stop()


# ============================================================
# CHECK DATA
# ============================================================

if "song" not in df.columns:

    st.error(
        "❌ Your songs.csv file must contain a 'song' column."
    )

    st.stop()


if "artist" not in df.columns:

    st.error(
        "❌ Your songs.csv file must contain an 'artist' column."
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

def recommend_songs(song_name, number_of_songs=5):

    songs_lower = (
        df["song"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    matches = df[
        songs_lower == song_name.strip().lower()
    ]

    if matches.empty:
        return None

    selected_index = matches.index[0]

    matrix_index = df.index.get_loc(selected_index)

    scores = cosine_similarity(
        tfidf_matrix[matrix_index],
        tfidf_matrix
    ).flatten()

    sorted_indices = scores.argsort()[::-1]

    sorted_indices = [
        index
        for index in sorted_indices
        if index != matrix_index
    ]

    sorted_indices = sorted_indices[:number_of_songs]

    result = df.iloc[
        sorted_indices
    ][["artist", "song"]]

    return result.reset_index(drop=True)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-box">',
    unsafe_allow_html=True
)

st.markdown("# 🎧 TuneMatch")

st.markdown(
    "### Discover music that matches your vibe."
)

st.caption(
    "✨ Powered by Machine Learning"
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# SEARCH SECTION
# ============================================================

st.markdown(
    '<div class="search-box">',
    unsafe_allow_html=True
)

st.subheader("🎵 Find Your Next Favorite Song")

st.write(
    "Select a song and TuneMatch will find tracks "
    "with a similar musical vibe."
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


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
            number_of_songs=5
        )

        if recommendations is None:

            st.error(
                "❌ Song not found."
            )

        else:

            st.markdown("## ✨ Your Recommendations")

            st.write(
                f"Songs similar to **{selected_song}**"
            )

            for i, row in recommendations.iterrows():

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                col_a, col_b, col_c = st.columns(
                    [0.7, 5, 0.7]
                )

                with col_a:

                    st.markdown(
                        f"### 🎵"
                    )

                with col_b:

                    st.markdown(
                        f"**#{i + 1} — {row['song']}**"
                    )

                    st.caption(
                        f"🎤 {row['artist']}"
                    )

                with col_c:

                    st.markdown("▶️")

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "🎧 **TuneMatch**"
)

st.caption(
    "Music Recommendation System"
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)
