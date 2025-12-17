import pandas as pd
import numpy as np

def build_user_item_matrix(ratings):
    """
    Rows: Users
    Columns: Movies
    Values: Ratings
    """
    user_item_matrix = ratings.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    )

    return user_item_matrix


def calculate_sparsity(matrix):
    total_cells = matrix.shape[0] * matrix.shape[1]
    filled_cells = matrix.count().sum()
    sparsity = 1 - (filled_cells / total_cells)

    return sparsity
