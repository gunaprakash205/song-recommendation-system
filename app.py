import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="TuneMatch 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(138,43,226,0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(255,20,147,0.15), transparent 30%),
        linear-gradient(135deg, #090014, #12001f 45%, #08000f);
    color: white;
}

/* Remove top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */
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
    background: linear-gradient(90deg, #ff4ecd, #a855f7, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 18px;
    color: #b9aec7;
    margin-top: 10px;
}

/* Search container */
.search-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 25px;
    margin: 20px auto 35px;
    max-width: 900px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}

/* Recommendation cards */
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

    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

.card:hover {
    transform: translateY(-4px);
    border-color: rgba(168,85,247,0.6);
    box-shadow: 0 12px 35px rgba(168,85,247,0.18);
}

.song-number {
    font-size: 15px;
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

/* Section title */
.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #776d82;
    font-size: 13px;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("songs.csv")

    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)

    return df, cosine_sim


df, cosine_sim = load_data()


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_songs(song_name, cosine_sim=cosine_sim, df=df, top_n=5):

    idx = df[
        df["song"].str.lower() == song_name.lower()
    ].index

    if len(idx) == 0:
        return None

    idx = idx[0]

    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n + 1]

    song_indices = [
        i[0] for i in sim_scores
    ]

    return df.iloc[song_indices]


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">

    <div class="hero-icon">🎧</div>

    <h1>TuneMatch</h1>

    <p>
        Discover your next favorite song using
        <b>AI-powered recommendations</b>
    </p>

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

st.markdown("""
<div class="search-box">
<h3 style="text-align:center;">
🎵 Find a Song You Love
</h3>
</div>
""", unsafe_allow_html=True)


song_list = sorted(
    df["song"].dropna().unique()
)

selected_song = st.selectbox(
    "Choose a song",
    song_list,
    index=None,
    placeholder="Search for a song..."
)


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    recommend_button = st.button(
        "✨  Recommend Songs",
        use_container_width=True
    )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if recommend_button:

    if selected_song is None:

        st.warning(
            "🎵 Please select a song first."
        )

    else:

        recommendations = recommend_songs(
            selected_song
        )

        if recommendations is None:

            st.error(
                "Song not found in the dataset."
            )

        else:

            st.markdown(
                f"""
                <div class="section-title">
                    🎶 Recommended for you
                </div>

                <p style="color:#aaa0b5;">
                    Because you liked
                    <b style="color:white;">
                    {selected_song}
                    </b>
                </p>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------
            # RECOMMENDATION CARDS
            # ------------------------------------------

            for i, (_, row) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="card">

                        <div style="display:flex;
                                    align-items:center;
                                    gap:18px;">

                            <div class="music-icon">
                                🎵
                            </div>

                            <div style="flex:1;">

                                <div class="song-number">
                                    #{i}
                                </div>

                                <div class="song-title">
                                    {row['song']}
                                </div>

                                <div class="artist">
                                    🎤 {row['artist']}
                                </div>

                            </div>

                            <div style="font-size:25px;">
                                ▶️
                            </div>

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

    🎧 TuneMatch • Music Recommendation System

    <br><br>

    Built with Python • Pandas • Scikit-learn • Streamlit

</div>
""", unsafe_allow_html=True)
