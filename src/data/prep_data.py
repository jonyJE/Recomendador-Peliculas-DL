import pandas as pd

def cargar_y_limpiar_datos(path_raw='../data/raw/'):
    """Carga los archivos de MovieLens y aplica Label Encoding."""
    # Carga de archivos
    ratings = pd.read_csv(f'{path_raw}ratings.dat', sep='::', 
                         names=['user_id', 'movie_id', 'rating', 'timestamp'], 
                         engine='python', encoding='latin-1')
    movies = pd.read_csv(f'{path_raw}movies.dat', sep='::', 
                        names=['movie_id', 'title', 'genres'], 
                        engine='python', encoding='latin-1')

    # Label Encoding (Convertir IDs a índices 0...N)
    ratings['user'] = ratings['user_id'].astype('category').cat.codes
    ratings['movie'] = ratings['movie_id'].astype('category').cat.codes
    
    return ratings, movies

def calcular_sparsity(ratings):
    """Calcula el porcentaje de celdas vacías en la matriz."""
    n_users = ratings['user'].nunique()
    n_items = ratings['movie'].nunique()
    n_ratings = len(ratings)
    sparsity = (1 - n_ratings / (n_users * n_items)) * 100
    return sparsity