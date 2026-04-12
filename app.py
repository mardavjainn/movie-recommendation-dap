"""
🎬 Movie Recommendation System — Streamlit Web Application
==========================================================
Dataset  : TMDB Movies Dataset 2023 (930K+ Movies)
ML Model : Content-Based Filtering (CountVectorizer + Cosine Similarity)
API      : TMDB API for posters & metadata
Author   : Data Analytics Project (DAP)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TMDB_API_KEY = "f882762ab569a7b595b61a89a3fab14a"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
FALLBACK_POSTER = "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
ARTIFACTS_DIR = Path("artifacts")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 MovieLens AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — modern dark theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #30363d;
    text-align: center;
}
.hero-banner h1 { font-size: 2.6rem; margin: 0; color: #e2c27d; }
.hero-banner p  { font-size: 1.05rem; color: #9daabb; margin-top: 0.4rem; }

/* ── Movie card ── */
.movie-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    border-color: #e2c27d;
}
.movie-card img {
    width: 100%;
    border-radius: 8px;
    object-fit: cover;
    height: 280px;
}
.movie-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2c27d;
    margin: 0.5rem 0 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-meta {
    font-size: 0.78rem;
    color: #8b949e;
}
.badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #58a6ff;
    margin: 2px;
}
.rating-star { color: #f0c040; }

/* ── Section header ── */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2c27d;
    border-left: 4px solid #e2c27d;
    padding-left: 0.7rem;
    margin: 1.5rem 0 1rem;
}

/* ── Input & button ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e0e0e0;
}
.stButton > button {
    background: linear-gradient(90deg, #e2c27d, #d4a843);
    color: #0e1117;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 2.5rem;
    border-radius: 8px;
    border: none;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Trending card ── */
.trend-card {
    background: #161b22;
    border-radius: 10px;
    padding: 0.6rem;
    border: 1px solid #30363d;
    text-align: center;
}
.trend-card img { width:100%; height:160px; object-fit:cover; border-radius:6px; }
.trend-title { font-size:0.8rem; color:#e2c27d; margin-top:0.4rem; font-weight:600;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.trend-rating { font-size:0.75rem; color:#8b949e; }

/* ── Overview box ── */
.overview-box {
    background: #161b22;
    border-left: 3px solid #e2c27d;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 0.4rem;
}

/* ── Metric chips ── */
.metric-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.8rem; color: #c9d1d9; margin: 3px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load pre-built movie dictionary and similarity matrix from artifacts/."""
    movie_dict_path = ARTIFACTS_DIR / "movie_dict.pkl"
    similarity_path = ARTIFACTS_DIR / "similarity.pkl"

    if not movie_dict_path.exists() or not similarity_path.exists():
        return None, None

    with open(movie_dict_path, "rb") as f:
        movie_dict = pickle.load(f)
    with open(similarity_path, "rb") as f:
        similarity = pickle.load(f)

    movies_df = pd.DataFrame(movie_dict)
    return movies_df, similarity


# ──────────────────────────────────────────────────────────────────────────────
# TMDB API HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_tmdb_data(movie_title: str, year: int = None):
    """
    Fetch poster URL, overview, and rating from TMDB by movie title.
    Returns dict with keys: poster_url, overview, rating, tmdb_id
    """
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_title,
            "language": "en-US",
            "page": 1,
        }
        if year and not np.isnan(year):
            params["year"] = int(year)

        resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:
            # Retry without year filter
            params.pop("year", None)
            resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
            results = resp.json().get("results", [])

        if results:
            r = results[0]
            poster_path = r.get("poster_path")
            return {
                "poster_url": TMDB_POSTER_BASE + poster_path if poster_path else FALLBACK_POSTER,
                "overview":   r.get("overview", ""),
                "rating":     r.get("vote_average", 0.0),
                "tmdb_id":    r.get("id"),
            }
    except Exception:
        pass

    return {"poster_url": FALLBACK_POSTER, "overview": "", "rating": 0.0, "tmdb_id": None}


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_trending():
    """Fetch trending movies from TMDB."""
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        return resp.json().get("results", [])[:10]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def get_recommendations(selected_title: str, movies_df: pd.DataFrame,
                         similarity: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """
    Return top_n recommended movies for selected_title.
    Uses cosine similarity scores from pre-computed matrix.
    """
    titles = movies_df["title"].tolist()
    idx = next((i for i, t in enumerate(titles) if t == selected_title), None)
    if idx is None:
        return pd.DataFrame()

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    rec_indices = [s[0] for s in sim_scores]
    rec_similarity = [round(s[1], 4) for s in sim_scores]

    recs = movies_df.iloc[rec_indices].copy().reset_index(drop=True)
    recs["similarity_score"] = rec_similarity
    return recs


# ──────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
def render_movie_card(title: str, year, rating, poster_url: str,
                      overview: str = "", similarity: float = None):
    """Render a single movie card as HTML."""
    year_str = str(int(year)) if year and not (isinstance(year, float) and np.isnan(year)) else "N/A"
    rating_str = f"⭐ {rating:.1f}" if rating else "N/A"
    sim_str = f"<br><span class='badge'>🔗 {similarity:.0%} match</span>" if similarity else ""
    overview_trunc = (overview[:120] + "…") if len(overview) > 120 else overview
    overview_html = f"<div class='overview-box'>{overview_trunc}</div>" if overview_trunc else ""

    return f"""
    <div class='movie-card'>
        <img src='{poster_url}' alt='{title}' onerror="this.src='{FALLBACK_POSTER}'" />
        <div class='movie-title' title='{title}'>{title}</div>
        <div class='movie-meta'>
            <span class='metric-chip'>📅 {year_str}</span>
            <span class='metric-chip rating-star'>{rating_str}</span>
            {sim_str}
        </div>
        {overview_html}
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar(movies_df):
    with st.sidebar:
        st.markdown("### 🎛️ Filters")

        # Genre filter
        all_genres = set()
        for g in movies_df["genres"].dropna():
            if isinstance(g, str):
                for genre in g.replace(",", " ").split():
                    if len(genre) > 2:
                        all_genres.add(genre.strip().capitalize())
        genres_sorted = ["All"] + sorted(all_genres)
        selected_genre = st.selectbox("🎭 Genre", genres_sorted)

        # Year range
        years = movies_df["year"].dropna().astype(int)
        min_y, max_y = int(years.min()), int(years.max())
        year_range = st.slider("📅 Release Year", min_y, max_y, (max(min_y, 1990), max_y))

        # Minimum rating
        min_rating = st.slider("⭐ Min Rating", 0.0, 10.0, 5.0, 0.5)

        # Number of recommendations
        top_n = st.slider("🔢 Recommendations", 5, 20, 10)

        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        st.metric("Total Movies (ML)", f"{len(movies_df):,}")
        avg_r = movies_df["vote_average"].mean()
        st.metric("Avg Rating", f"{avg_r:.2f} ⭐")

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#6e7681;'>Powered by TMDB API · "
            "Content-Based Filtering · Cosine Similarity</div>",
            unsafe_allow_html=True,
        )

    return selected_genre, year_range, min_rating, top_n


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hero banner ──
    st.markdown("""
    <div class='hero-banner'>
        <h1>🎬 MovieLens AI</h1>
        <p>Content-Based Movie Recommendation System · TMDB 930K+ Movies · ML-Powered</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ──
    movies_df, similarity = load_artifacts()

    if movies_df is None:
        st.error("⚠️ Artifacts not found. Please run `notebook.ipynb` first to generate `artifacts/movie_dict.pkl` and `artifacts/similarity.pkl`.")
        st.info(
            "**Steps to fix:**\n"
            "1. Download TMDB dataset from Kaggle\n"
            "2. Place `TMDB_movie_dataset_v11.csv` in the project folder\n"
            "3. Run all cells in `notebook.ipynb`\n"
            "4. Restart this Streamlit app"
        )

        # Demo mode with TMDB trending data
        st.markdown("<div class='section-header'>🔥 Trending Movies (Demo Mode)</div>", unsafe_allow_html=True)
        trending = fetch_trending()
        if trending:
            cols = st.columns(5)
            for i, movie in enumerate(trending[:5]):
                with cols[i]:
                    poster = TMDB_POSTER_BASE + movie.get("poster_path", "") if movie.get("poster_path") else FALLBACK_POSTER
                    st.image(poster, use_column_width=True)
                    st.caption(f"**{movie.get('title', 'Unknown')}**\n⭐ {movie.get('vote_average', 0):.1f}")
        return

    # ── Sidebar ──
    selected_genre, year_range, min_rating, top_n = render_sidebar(movies_df)

    # ── Apply filters for dropdown ──
    filtered = movies_df.copy()
    if selected_genre != "All":
        filtered = filtered[
            filtered["genres"].str.contains(selected_genre, case=False, na=False)
        ]
    filtered = filtered[
        (filtered["year"].fillna(0).astype(int) >= year_range[0]) &
        (filtered["year"].fillna(9999).astype(int) <= year_range[1])
    ]
    filtered = filtered[filtered["vote_average"] >= min_rating]
    filtered = filtered.sort_values("popularity", ascending=False)

    movie_titles = filtered["title"].dropna().tolist()

    if not movie_titles:
        st.warning("No movies match the current filters. Try widening your search.")
        return

    # ── Movie search ──
    st.markdown("<div class='section-header'>🔍 Find Your Movie</div>", unsafe_allow_html=True)

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        selected_movie = st.selectbox(
            "Search or select a movie:",
            movie_titles,
            label_visibility="collapsed",
            placeholder="Type a movie name…",
        )
    with col_btn:
        recommend_btn = st.button("🎯 Recommend", use_container_width=True)

    # ── Selected movie info ──
    if selected_movie:
        row = movies_df[movies_df["title"] == selected_movie].iloc[0]
        tmdb_data = fetch_tmdb_data(
            selected_movie,
            row.get("year") if "year" in row.index else None,
        )

        col_img, col_info = st.columns([1, 3])
        with col_img:
            st.image(tmdb_data["poster_url"], width=200)
        with col_info:
            yr = str(int(row["year"])) if "year" in row.index and row["year"] and not np.isnan(float(row["year"])) else "N/A"
            st.markdown(f"### {selected_movie} ({yr})")
            rating = tmdb_data["rating"] or row.get("vote_average", 0)
            st.markdown(
                f"<span class='metric-chip rating-star'>⭐ {rating:.1f}/10</span>"
                f"<span class='metric-chip'>🎭 {row.get('genres', 'N/A')}</span>"
                f"<span class='metric-chip'>🔥 Popularity: {row.get('popularity', 0):.0f}</span>",
                unsafe_allow_html=True,
            )
            overview = tmdb_data["overview"] or row.get("overview", "")
            if overview:
                st.markdown(f"<div class='overview-box'>{overview}</div>", unsafe_allow_html=True)

    # ── Recommendations ──
    if recommend_btn and selected_movie:
        with st.spinner(f"🤖 Finding movies similar to **{selected_movie}**…"):
            recs = get_recommendations(selected_movie, movies_df, similarity, top_n)

        if len(recs) == 0:
            st.warning("Could not find recommendations. Try a different movie.")
        else:
            st.markdown(
                f"<div class='section-header'>🎯 Top {len(recs)} Recommendations for '{selected_movie}'</div>",
                unsafe_allow_html=True,
            )

            COLS_PER_ROW = 5
            rows = [recs.iloc[i:i+COLS_PER_ROW] for i in range(0, len(recs), COLS_PER_ROW)]

            for row_df in rows:
                cols = st.columns(COLS_PER_ROW)
                for col, (_, movie_row) in zip(cols, row_df.iterrows()):
                    tmdb = fetch_tmdb_data(
                        movie_row["title"],
                        movie_row.get("year"),
                    )
                    yr = movie_row.get("year")
                    rating = tmdb["rating"] or movie_row.get("vote_average", 0)
                    overview = tmdb.get("overview", "")
                    sim_score = movie_row.get("similarity_score")

                    with col:
                        st.markdown(
                            render_movie_card(
                                title=movie_row["title"],
                                year=yr,
                                rating=rating,
                                poster_url=tmdb["poster_url"],
                                overview=overview,
                                similarity=sim_score,
                            ),
                            unsafe_allow_html=True,
                        )

            # ── Recommendation explanation ──
            with st.expander("🧠 Why these movies were recommended"):
                st.markdown(
                    "These movies were selected based on **cosine similarity** computed over "
                    "a content vector that combines:\n"
                    "- 🎭 **Genres** (Action, Comedy, Drama…)\n"
                    "- 📝 **Plot Overview** (TF-IDF text representation)\n"
                    "- 🏷️ **Keywords** (if available)\n\n"
                    "Movies with higher similarity scores share more content features "
                    "with your selected movie."
                )
                st.dataframe(
                    recs[["title", "genres", "year", "vote_average", "similarity_score"]]
                    .rename(columns={
                        "title": "Title", "genres": "Genres", "year": "Year",
                        "vote_average": "Rating", "similarity_score": "Similarity"
                    }),
                    use_container_width=True,
                )

    # ── Trending section ──
    st.markdown("<div class='section-header'>🔥 Trending This Week</div>", unsafe_allow_html=True)
    trending = fetch_trending()

    if trending:
        trend_cols = st.columns(10)
        for i, movie in enumerate(trending[:10]):
            with trend_cols[i]:
                poster = (
                    TMDB_POSTER_BASE + movie["poster_path"]
                    if movie.get("poster_path")
                    else FALLBACK_POSTER
                )
                st.markdown(
                    f"""<div class='trend-card'>
                        <img src='{poster}' onerror="this.src='{FALLBACK_POSTER}'" />
                        <div class='trend-title' title='{movie.get("title","")}'>{movie.get("title","")}</div>
                        <div class='trend-rating'>⭐ {movie.get("vote_average",0):.1f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Could not load trending movies. Check your internet connection.")

    # ── Top rated from dataset ──
    st.markdown("<div class='section-header'>🏆 Top Rated in Dataset</div>", unsafe_allow_html=True)
    top_rated = (
        movies_df[movies_df["vote_average"] >= 7.0]
        .sort_values(["vote_average", "popularity"], ascending=False)
        .head(10)
    )
    tr_cols = st.columns(5)
    for i, (_, row) in enumerate(top_rated.head(5).iterrows()):
        tmdb = fetch_tmdb_data(row["title"], row.get("year"))
        with tr_cols[i]:
            st.image(tmdb["poster_url"], use_column_width=True)
            yr = str(int(row["year"])) if "year" in row.index and row["year"] and not np.isnan(float(row["year"])) else ""
            st.caption(f"**{row['title']}** ({yr})\n⭐ {tmdb['rating'] or row.get('vote_average', 0):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
