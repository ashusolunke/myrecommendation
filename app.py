import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# -------------------------------
# Load Movies Dataset
# -------------------------------
@st.cache_data(show_spinner=True)
def load_movies():
    movies = pd.read_csv("movies.csv")

    # Ensure required columns exist
    required_cols = ["movieId", "title", "genres"]
    for col in required_cols:
        if col not in movies.columns:
            st.error(f"❌ Column '{col}' missing in movies.csv. Found columns: {list(movies.columns)}")
            st.stop()

    # Clean genres (convert JSON-like to plain text if needed)
    def clean_genres(g):
        if isinstance(g, str) and g.startswith("["):
            import ast
            try:
                parsed = ast.literal_eval(g)
                if isinstance(parsed, list):
                    return " ".join(
                        [d["name"] for d in parsed if isinstance(d, dict) and "name" in d]
                    )
            except Exception:
                return g
        return g

    movies["genres"] = movies["genres"].fillna("").astype(str).apply(clean_genres)
    movies["title"] = movies["title"].astype(str)

    return movies


# -------------------------------
# TMDB API Functions
# -------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY") or "your_tmdb_api_key_here"

def fetch_poster(movie_title):
    """Fetch poster URL from TMDB API"""
    if TMDB_API_KEY == "your_tmdb_api_key_here":
        return None
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path", None)
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
    except:
        return None
    return None


# -------------------------------
# Recommendation System
# -------------------------------
@st.cache_data(show_spinner=True)
def build_tfidf(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend(movie_title, movies, cosine_sim, top_n=10):
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    if movie_title not in indices:
        return []
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="MyRecommendation", page_icon="🎬", layout="wide")

st.title("🎬 MyRecommendation")
st.write("Content-based movie recommender with posters (using TMDB API).")

movies = load_movies()
cosine_sim = build_tfidf(movies)

# Sidebar Settings
st.sidebar.header("⚙ Settings")
top_n = st.sidebar.slider("How many recommendations?", 5, 20, 10)

if TMDB_API_KEY and TMDB_API_KEY != "your_tmdb_api_key_here":
    st.sidebar.success("✅ TMDB key detected")
else:
    st.sidebar.warning("⚠ No TMDB key found. Posters may not load.")

# Select movie
title_series = movies["title"]
if isinstance(title_series, pd.DataFrame):
    title_series = title_series.iloc[:, 0]

selected_title = st.selectbox(
    "Pick a movie", 
    options=sorted(title_series.dropna().astype(str).unique().tolist())
)

if selected_title:
    st.subheader(f"🎥 Recommendations for **{selected_title}**")

    recs = recommend(selected_title, movies, cosine_sim, top_n=top_n)

    cols = st.columns(5)
    for idx, (_, row) in enumerate(recs.iterrows()):
        with cols[idx % 5]:
            poster_url = fetch_poster(row["title"])
            if poster_url:
                st.image(poster_url, use_column_width=True)
            st.caption(row["title"])
            st.write(f"**Genres:** {row['genres']}")
