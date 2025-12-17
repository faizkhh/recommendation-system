# similarity.py
import numpy as np
import pandas as pd

# -----------------------------
# Similarity Functions
# -----------------------------
def cosine_similarity(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if np.sum(mask) == 0:
        return 0
    a_filtered = a[mask]
    b_filtered = b[mask]
    num = np.dot(a_filtered, b_filtered)
    denom = np.linalg.norm(a_filtered) * np.linalg.norm(b_filtered)
    if denom == 0:
        return 0
    return num / denom

def pearson_correlation(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if np.sum(mask) == 0:
        return 0
    a_filtered = a[mask]
    b_filtered = b[mask]
    a_mean = np.mean(a_filtered)
    b_mean = np.mean(b_filtered)
    num = np.sum((a_filtered - a_mean) * (b_filtered - b_mean))
    denom = np.sqrt(np.sum((a_filtered - a_mean)**2)) * np.sqrt(np.sum((b_filtered - b_mean)**2))
    if denom == 0:
        return 0
    return num / denom

# -----------------------------
# Main Execution (Self-Contained)
# -----------------------------
if __name__ == "__main__":
    # Load dataset directly
    data_path = "data/u.data"  # Update this if your file is elsewhere
    ratings = pd.read_csv(data_path, sep='\t', names=['user_id','movie_id','rating','timestamp'])
    print("First 5 rows of ratings:")
    print(ratings.head())

    # Build user-item matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    print("User-Item Matrix shape:", user_item_matrix.shape)

    # Pick two users (make sure these IDs exist)
    user1 = user_item_matrix.loc[196].astype(float).values
    user2 = user_item_matrix.loc[186].astype(float).values

    # Compute similarity
    cos_sim = cosine_similarity(user1, user2)
    pearson_sim = pearson_correlation(user1, user2)

    print(f"Cosine similarity (user196, user186): {cos_sim:.4f}")
    print(f"Pearson correlation (user196, user186): {pearson_sim:.4f}")
