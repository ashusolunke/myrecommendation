import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Content-based: using genres
def content_based_recommender(movie_title, top_n=10):
    cv = CountVectorizer(stop_words="english")
    count_matrix = cv.fit_transform(movies["genres"].fillna(""))
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["movieId", "title", "genres"]]

# Collaborative filtering (user-based simple)
def collaborative_recommender(user_id, top_n=10):
    user_movie_ratings = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    user_movie_ratings = user_movie_ratings.fillna(0)
    
    similarity = cosine_similarity(user_movie_ratings)
    sim_df = pd.DataFrame(similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)
    
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:6].index
    recommended_movies = ratings[ratings["userId"].isin(similar_users)]
    top_movies = recommended_movies.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(top_n)
    
    return movies[movies["movieId"].isin(top_movies.index)][["movieId", "title", "genres"]]

# Hybrid (combine both)
def hybrid_recommender(movie_title, user_id, top_n=10):
    content_recs = content_based_recommender(movie_title, top_n)
    collab_recs = collaborative_recommender(user_id, top_n)
    final_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().head(top_n)
    return final_recs