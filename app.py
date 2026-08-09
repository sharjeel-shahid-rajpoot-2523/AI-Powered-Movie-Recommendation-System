import ast
import os
import pickle

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Data building (runs once, then cached) ----------

def convert(obj):
    """Extract the 'name' field from a JSON-like list of dicts."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


@st.cache_resource
def load_data():
    """
    Build (or load) movies dataframe + cosine similarity matrix.

    If movie_data.pkl already exists on disk, load it directly (fast path).
    Otherwise, rebuild it from the raw CSVs (tmdb_5000_credits.csv and
    tmdb_5000_movies.csv), which live in the repo, and cache the pickle
    for next time.
    """
    pkl_path = 'movie_data.pkl'

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            movies, cosine_sim = pickle.load(file)
        return movies, cosine_sim

    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')

    movies = movies.merge(credits, left_on='title', right_on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
    movies['crew'] = movies['crew'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director']
    )

    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies = movies[['movie_id', 'title', 'overview', 'tags']]
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Cache to disk so subsequent app restarts skip rebuilding
    with open(pkl_path, 'wb') as file:
        pickle.dump((movies, cosine_sim), file)

    return movies, cosine_sim


movies, cosine_sim = load_data()


# ---------- Recommendation logic ----------

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'movie_id']].iloc[movie_indices]


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API. Returns None if unavailable."""
    api_key = '7b995d3c6fd91a2284b4ad8cb390c7b8'  # Replace with your TMDB API key
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.RequestException:
        return None


# ---------- Streamlit UI ----------

st.title("Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button('Recommend'):
    recommendations = get_recommendations(selected_movie)
    st.write("Top 10 recommended movies:")

    # Create a 2x5 grid layout
    for i in range(0, 10, 5):  # Loop over rows (2 rows, 5 movies each)
        cols = st.columns(5)  # Create 5 columns for each row
        for col, j in zip(cols, range(i, i + 5)):
            if j < len(recommendations):
                movie_title = recommendations.iloc[j]['title']
                movie_id = recommendations.iloc[j]['movie_id']
                poster_url = fetch_poster(movie_id)
                with col:
                    if poster_url:
                        st.image(poster_url, width=130)
                    else:
                        st.write("No poster available")
                    st.write(movie_title)
