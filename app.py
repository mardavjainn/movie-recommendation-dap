import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path

# ───────────────────────── CONFIG ─────────────────────────
TMDB_API_KEY = "YOUR_API_KEY"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
FALLBACK_POSTER = "https://via.placeholder.com/300x450?text=No+Image"
ARTIFACTS_DIR = Path("artifacts")

# ───────────────────────── PAGE ─────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide"
)

# ───────────────────────── NETFLIX STYLE CSS ─────────────────────────
st.markdown("""
<style>

/* Background */
body, .stApp {
    background-color: #ffffff;
    color: #111111;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e5e7eb;
}

/* Title */
h1 {
    color: #e50914;
}

/* Buttons */
.stButton > button {
    background-color: #e50914;
    color: white;
    border-radius: 6px;
    padding: 10px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #b20710;
}

/* Cards */
.card {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e5e7eb;
    transition: 0.2s;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.1);
}

/* Image */
.card img {
    border-radius: 8px;
    width: 100%;
}

/* Text */
.card h4 {
    font-size: 0.95rem;
    margin-top: 8px;
}
.small {
    font-size: 0.8rem;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)

# ───────────────────────── LOAD DATA ─────────────────────────
@st.cache_data
def load_data():
    with open(ARTIFACTS_DIR / "movie_dict.pkl", "rb") as f:
        movie_dict = pickle.load(f)

    with open(ARTIFACTS_DIR / "similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    return pd.DataFrame(movie_dict), similarity

# ───────────────────────── TMDB API ─────────────────────────
@st.cache_data
def fetch_movie(title):
    try:
        params = {"api_key": TMDB_API_KEY, "query": title}
        res = requests.get(TMDB_SEARCH_URL, params=params).json()

        if res["results"]:
            movie = res["results"][0]
            poster = movie.get("poster_path")
            return {
                "poster": TMDB_POSTER_BASE + poster if poster else FALLBACK_POSTER,
                "rating": movie.get("vote_average", 0)
            }
    except:
        pass

    return {"poster": FALLBACK_POSTER, "rating": 0}

# ───────────────────────── RECOMMENDATION ─────────────────────────
def recommend(movie, df, similarity):
    index = df[df["title"] == movie].index[0]
    distances = similarity[index]
    movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    result = []
    for i in movies:
        result.append(df.iloc[i[0]])
    return pd.DataFrame(result)

# ───────────────────────── CARD UI ─────────────────────────
def movie_card(title, poster, rating):
    return f"""
    <div class="card">
        <img src="{poster}">
        <h4>{title}</h4>
        <p class="small">⭐ {rating}</p>
    </div>
    """

# ───────────────────────── MAIN APP ─────────────────────────
def main():
    # Header
    st.markdown("<h1>🎬 Movie Recommender</h1>", unsafe_allow_html=True)
    st.caption("Discover movies you’ll love")

    df, similarity = load_data()

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        genres = ["All"] + sorted(set(" ".join(df["genres"].dropna()).split()))
        selected_genre = st.selectbox("Genre", genres)

        years = df["year"].dropna().astype(int)
        year_range = st.slider("Year", int(years.min()), int(years.max()), (2000, int(years.max())))

    # Filtering
    filtered = df.copy()

    if selected_genre != "All":
        filtered = filtered[filtered["genres"].str.contains(selected_genre, case=False, na=False)]

    filtered = filtered[
        (filtered["year"] >= year_range[0]) &
        (filtered["year"] <= year_range[1])
    ]

    movie_list = filtered["title"].dropna().tolist()

    # Select movie
    selected_movie = st.selectbox("Select a movie", movie_list)

    if st.button("Recommend"):
        recs = recommend(selected_movie, df, similarity)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Recommended for you")

        cols = st.columns(5)

        for i, (_, row) in enumerate(recs.iterrows()):
            data = fetch_movie(row["title"])

            with cols[i % 5]:
                st.markdown(
                    movie_card(
                        row["title"],
                        data["poster"],
                        data["rating"]
                    ),
                    unsafe_allow_html=True
                )

# ───────────────────────── RUN ─────────────────────────
if __name__ == "__main__":
    main()