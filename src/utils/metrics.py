import numpy as np

def calcular_rmse(model, test_df):
    """Calcula el error promedio en estrellas."""
    predictions = model.predict([test_df['user'], test_df['movie']], verbose=0)
    # Volvemos a escala de 1-5 estrellas
    predictions_stars = predictions.flatten() * 5
    real_stars = test_df['rating'].values
    
    rmse = np.sqrt(np.mean((predictions_stars - real_stars)**2))
    return rmse

def obtener_recomendaciones(model, ratings_df, movies_df, user_id, top_n=5):
    """Genera una lista de películas recomendadas para un usuario."""
    # Datos del usuario
    user_index = ratings_df[ratings_df['user_id'] == user_id]['user'].iloc[0]
    
    # Filtrar lo que no ha visto
    vistas = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique()
    todas = ratings_df['movie_id'].unique()
    por_ver = [m for m in todas if m not in vistas]
    
    # Mapear a índices de la IA
    indices_por_ver = ratings_df[ratings_df['movie_id'].isin(por_ver)]['movie'].unique()
    
    # Preparar inputs para el modelo
    u_input = np.array([user_index] * len(indices_por_ver)).reshape(-1, 1)
    m_input = indices_por_ver.reshape(-1, 1)
    
    # Predecir
    preds = model.predict([u_input, m_input], verbose=0)
    
    # Obtener top N
    top_idx = preds.flatten().argsort()[-top_n:][::-1]
    final_movie_indices = indices_por_ver[top_idx]
    
    # Traducir a títulos
    res = []
    for m_idx in final_movie_indices:
        real_id = ratings_df[ratings_df['movie'] == m_idx]['movie_id'].iloc[0]
        title = movies_df[movies_df['movie_id'] == real_id]['title'].values[0]
        res.append(title)
        
    return res