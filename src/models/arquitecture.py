import tensorflow as tf
from tensorflow.keras import layers, models

def crear_modelo_ncf(n_users, n_items, embedding_size=50):
    """Define la arquitectura Neural Collaborative Filtering."""
    # Entradas
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')

    # Embeddings
    user_embedding = layers.Embedding(n_users, embedding_size, name='user_emb')(user_input)
    movie_embedding = layers.Embedding(n_items, embedding_size, name='movie_emb')(movie_input)

    # Aplanar vectores
    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)

    # Capas Densas (MLP)
    prod = layers.Dot(axes=1)([user_vec, movie_vec])
    dense = layers.Dense(64, activation='relu')(prod)
    dense = layers.Dense(32, activation='relu')(dense)
    output = layers.Dense(1, activation='sigmoid')(dense)

    # Compilación
    model = models.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model