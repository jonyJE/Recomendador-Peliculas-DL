# 🎬 Sistema de Recomendación de Películas con Deep Learning

Este proyecto implementa un motor de recomendación basado en **Neural Collaborative Filtering (NCF)** utilizando el dataset MovieLens 1M. El sistema es capaz de predecir la calificación que un usuario le daría a una película basándose en sus gustos previos y en los patrones de otros usuarios.

## 🚀 Estructura del Proyecto
- `data/`: Contiene los archivos originales y procesados.
- `notebooks/`: Análisis exploratorio y entrenamiento del modelo.
- `src/`: Código fuente para el procesamiento y arquitectura.
- `models/`: Pesos guardados del modelo entrenado.

## 📊 Análisis de Datos (EDA)
Encontramos que el dataset tiene una **dispersión (sparsity) del [PON AQUÍ TU 95.53]%**, lo cual representa un reto ideal para técnicas de Deep Learning. Las películas más populares incluyen títulos como *American Beauty* y *Star Wars*.

## 🧠 Arquitectura del Modelo
El modelo utiliza **Embeddings** para representar a usuarios e ítems en un espacio latente de 50 dimensiones, seguido de capas densas para capturar interacciones complejas.

## 📈 Resultados
- **Métrica de Error (RMSE):** [PON AQUÍ TU ERROR] estrellas.
- **Funcionalidad:** El sistema genera un Top 5 de recomendaciones personalizadas para cualquier usuario del sistema.

## 🛠️ Instalación
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar el notebook en la carpeta `notebooks/`.