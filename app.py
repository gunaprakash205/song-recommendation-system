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
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_data():

    songs = pd.read_csv("songs.csv")

    matrix = load_npz("tfidf_matrix.npz")

    return songs, matrix


# ============================================================
# LOAD FILES
# ============================================================

try:

    df, tfidf_matrix = load_data()

except pd.errors.EmptyDataError:

    st.error("❌ songs.csv is empty.")
    st.stop()

except FileNotFoundError as e:

    st.error(
        f"❌ Required file not found: {e.filename}"
    )
    st.info(
        "Make sure songs.csv and tfidf_matrix.npz "
        "are in the same GitHub repository as app.py."
    )
    st.stop()

except Exception as e:

    st.error(
        f"❌ Could not load the data: {str(e)}"
    )
    st.stop()


# ============================================================
# FIX ARTIST COLUMN
# ============================================================

# Your CSV currently appears to use "ist"
# instead of "artist".

if "artist" not in df.columns:

    if "ist" in df.columns:

        df = df.rename(
            columns={"ist": "artist"}
        )

    else:

        st.error(
            "❌ Artist column not found in songs.csv."
        )

        st.write(
            "Columns found:",
            list(df.columns)
        )

        st.stop()


# ============================================================
# CHECK SONG COLUMN
# ============================================================

if "song" not in df.columns:

    st.error(
        "❌ 'song' column was not found in songs.csv."
    )

    st.write(
        "Columns found:",
        list(df.columns)
    )

    st.stop()


# ============================================================
# CHECK MATRIX
# ============================================================

if len(df) != tfidf_matrix.shape[0]:

    st.error(
        "❌ Dataset mismatch!"
    )

    st.write(
        f"Songs in CSV: {len(df)}"
    )

    st.write(
        f"Rows in TF-IDF matrix: {tfidf_matrix.shape[0]}"
    )

    st.info(
        "songs.csv and tfidf_matrix.npz must be generated "
        "from the same dataset and contain the same number "
        "of songs."
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

    matrix_index = df.index.get_loc(
        selected_index
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

    similar_indices = similar_indices[
        :number_of_songs
    ]

    recommendations = df.iloc[
        similar_indices
    ][["artist", "song"]]

    return recommendations.reset_index(
        drop=True
    )


# ============================================================
# TITLE
# ============================================================

st.title("🎧 TuneMatch")

st.subheader(
    "Discover music that matches your vibe."
)

st.caption(
    "✨ Machine Learning Powered Music Recommendation"
)


# ============================================================
# SEARCH
# ============================================================

st.divider()

st.header("🎵 Find Your Next Favorite Song")

st.write(
    "Choose a song and TuneMatch will recommend "
    "similar songs based on their lyrics."
)


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
    "Choose a song",
    song_list,
    index=None,
    placeholder="🔎 Search for a song..."
)


# ============================================================
# RECOMMEND BUTTON
# ============================================================

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

        elif recommendations.empty:

            st.warning(
                "No similar songs were found."
            )

        else:

            st.divider()

            st.header("✨ Your Recommendations")

            st.write(
                f"Songs similar to **{selected_song}**"
            )

            for i, row in recommendations.iterrows():

                st.markdown(
                    f"### 🎵 #{i + 1} — {row['song']}"
                )

                st.write(
                    f"🎤 Artist: **{row['artist']}**"
                )

                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🎧 TuneMatch — Music Recommendation System"
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)
