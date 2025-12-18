import numpy as np
import pandas as pd

print("SVD Recommender started...")

# -----------------------------
# Load data
# -----------------------------
ratings = pd.read_csv(
    "data/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/u.item",
    sep="|",
    encoding="latin-1",
    names=[
        "movie_id", "title", "release_date", "video_release",
        "imdb_url", "unknown", "Action", "Adventure", "Animation",
        "Children", "Comedy", "Crime", "Documentary", "Drama",
        "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
)

print("Data loaded")

# -----------------------------
# User-Item Matrix
# -----------------------------
user_item_matrix = ratings.pivot(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

print("Matrix shape:", user_item_matrix.shape)

# -----------------------------
# Mean centering
# -----------------------------
R = user_item_matrix.values
user_mean = np.mean(R, axis=1)
R_demeaned = R - user_mean.reshape(-1, 1)

# -----------------------------
# Truncated SVD (FAST)
# -----------------------------
k = 20   # keep it small for speed
U, sigma, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
sigma = np.diag(sigma[:k])
U = U[:, :k]
Vt = Vt[:k, :]

print("SVD completed")

# -----------------------------
# Predictions
# -----------------------------
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_mean.reshape(-1, 1)

preds_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

# -----------------------------
# Recommend function
# -----------------------------
def recommend_movies(user_id, n=5):
    user_preds = preds_df.loc[user_id]
    watched = ratings[ratings.user_id == user_id].movie_id.tolist()

    recommendations = (
        user_preds.drop(watched)
        .sort_values(ascending=False)
        .head(n)
    )

    print(f"\nTop {n} recommendations for User {user_id}:\n")

    for i, movie_id in enumerate(recommendations.index, 1):
        title = movies.loc[movies.movie_id == movie_id, "title"].values[0]
        print(f"{i}. {title}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    recommend_movies(196)
