import pandas as pd

def load_ratings(path):
    ratings = pd.read_csv(
        path,
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    return ratings


def load_movies(path):
    movies = pd.read_csv(
        path,
        sep='|',
        encoding='latin-1',
        names=[
            'movie_id', 'title', 'release_date', 'video_release',
            'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
            'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
            'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
    )
    return movies[['movie_id', 'title']]
