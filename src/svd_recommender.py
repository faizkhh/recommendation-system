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
# Build user-item matrix
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
# Truncated SVD
# -----------------------------
k = 50  # increase components for better accuracy
U, sigma, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
sigma = np.diag(sigma[:k])
U = U[:, :k]
Vt = Vt[:k, :]

print("SVD completed")

# -----------------------------
# Predicted ratings
# -----------------------------
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_mean.reshape(-1, 1)
preds_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

# Clip ratings to 1-5
preds_df = preds_df.clip(1, 5)

# -----------------------------
# Recommend movies function
# -----------------------------
def recommend_movies_svd(user_id, movies_df, preds_df, n_recommendations=5):
    user_preds = preds_df.loc[user_id]
    watched = ratings[ratings.user_id == user_id].movie_id.tolist()

    recommendations = (
        user_preds.drop(watched)
        .sort_values(ascending=False)
        .head(n_recommendations)
    )

    results = []
    for movie_id, rating in recommendations.items():
        title = movies_df.loc[movies_df.movie_id == movie_id, "title"].values[0]
        results.append((title, rating))

    return results

# -----------------------------
# RMSE evaluation
# -----------------------------
def calculate_rmSE(preds_df, ratings_df):
    actual = []
    predicted = []

    for row in ratings_df.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id
        rating = row.rating

        pred_rating = preds_df.loc[user_id, movie_id]

        actual.append(rating)
        predicted.append(pred_rating)

    actual = np.array(actual)
    predicted = np.array(predicted)

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    user_id = 196
    recommendations = recommend_movies_svd(user_id, movies, preds_df, n_recommendations=5)

    print(f"\nTop 5 recommendations for User {user_id}:\n")
    for i, (movie, rating) in enumerate(recommendations, 1):
        print(f"{i}. {movie} → Predicted Rating: {rating:.2f}")

    # Calculate RMSE
    rmse = calculate_rmSE(preds_df, ratings)
    print(f"\nRMSE of SVD Model: {rmse:.4f}")
