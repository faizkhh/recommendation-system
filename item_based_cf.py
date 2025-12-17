import numpy as np
import pandas as pd

print("Loading data...")

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

user_item = ratings.pivot(index="user_id", columns="movie_id", values="rating")

print("Data loaded successfully")
print("User-Item matrix shape:", user_item.shape)


# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() < 2:
        return 0
    a, b = a[mask], b[mask]
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# Recommendation
# -----------------------------
def recommend(user_id, n=5):
    print(f"\nGenerating recommendations for user {user_id}...")

    user_ratings = user_item.loc[user_id].dropna()

    # LIMIT to top 20 rated movies (speed boost)
    user_ratings = user_ratings.sort_values(ascending=False).head(20)

    scores = {}

    for movie_id in user_item.columns:
        if movie_id in user_ratings.index:
            continue

        num, den = 0, 0

        for rated_movie, rating in user_ratings.items():
            sim = cosine_similarity(
                user_item[movie_id].values,
                user_item[rated_movie].values
            )

            if sim != 0:
                num += sim * rating
                den += abs(sim)

        if den > 0:
            scores[movie_id] = num / den

    if not scores:
        print("⚠ No recommendations found")
        return []

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    results = []
    for movie_id, rating in top:
        title = movies[movies.movie_id == movie_id].title.values[0]
        results.append((title, rating))

    return results


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    user_id = 196

    recommendations = recommend(user_id)

    print(f"\nTop Recommendations for User {user_id}:\n")

    for i, (movie, rating) in enumerate(recommendations, 1):
        print(f"{i}. {movie} → Predicted Rating: {rating:.2f}")
