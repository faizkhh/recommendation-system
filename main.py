from src.data_loader import load_ratings, load_movies
from src.matrix_builder import build_user_item_matrix, calculate_sparsity

ratings = load_ratings('data/u.data')
movies = load_movies('data/u.item')

user_item_matrix = build_user_item_matrix(ratings)
sparsity = calculate_sparsity(user_item_matrix)

print("Ratings shape:", ratings.shape)
print("User-Item Matrix shape:", user_item_matrix.shape)
print(f"Sparsity: {sparsity:.2%}")
