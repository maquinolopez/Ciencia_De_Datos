# -*- coding: utf-8 -*-
"""
PRÁCTICA GUIADA: CLUSTERING ESPECTRAL
=====================================
CÓDIGO COMPLETO PARA EJECUTAR EN SPYDER

Instrucciones:
1. Ejecuta todo el código (F5)
2. Modifica los parámetros en la sección "CONFIGURACIÓN"
3. Vuelve a ejecutar para ver los cambios
"""

# ==========================
# BLOQUE 1: IMPORTAR LIBRERÍAS
# ==========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

print(" Librerías importadas correctamente")

# ==========================
# BLOQUE 2: CONFIGURACIÓN - ¡MODIFICA ESTOS PARÁMETROS!
# ==========================
#  PARÁMETROS PARA EXPERIMENTAR:

# Dataset
dataset_type = "circles"      # CAMBIA: "moons" o "circles"
k = 3                       # CAMBIA: número de clusters (2 o 3)
noise = .05                # CAMBIA: ruido (0.01 a 0.15)
n_samples = 200             # CAMBIA: número de puntos

# Spectral Clustering  
affinity_type = 'nearest_neighbors'       # CAMBIA: 'rbf' o 'nearest_neighbors'
gamma_value = 1.0          # CAMBIA: para 'rbf' (1, 10, 20, 50)
n_neighbors = 10            # CAMBIA: para 'nearest_neighbors' (5, 10, 15)

print(" PARÁMETROS ACTUALES:")
print(f"   Dataset: {dataset_type}")
print(f"   Clusters: {k}")
print(f"   Ruido: {noise}")
print(f"   Afinidad: {affinity_type}")
print(f"   Gamma: {gamma_value}")
print(f"   Vecinos: {n_neighbors}")

# ==========================
# BLOQUE 3: GENERAR DATASET
# ==========================
print("\n GENERANDO DATASET...")

if dataset_type == "moons":
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    print(" Dataset: 'Dos Lunas' generado")
    
elif dataset_type == "circles":
    X, y_true = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    print(" Dataset: 'Círculos Concéntricos' generado")

print(f" Forma del dataset: {X.shape}")

# ==========================
# BLOQUE 4: K-MEANS (MÉTODO LINEAL)
# ==========================
print("\n APLICANDO K-MEANS...")

km = KMeans(n_clusters=k, n_init=10, random_state=42)
labels_km = km.fit_predict(X)

silhouette_km = silhouette_score(X, labels_km)
print(f" Score de Silueta (K-means): {silhouette_km:.3f}")

# ==========================
# BLOQUE 5: CLUSTERING ESPECTRAL
# ==========================
print("\n APLICANDO CLUSTERING ESPECTRAL...")

# SOLO CAMBIA LAS VARIABLES affinity_type, gamma_value, n_neighbors EN LA SECCIÓN DE CONFIGURACIÓN
# El código automáticamente usará los parámetros correctos

spec = SpectralClustering(
    n_clusters=k,
    affinity=affinity_type,
    gamma=gamma_value if affinity_type == 'rbf' else 1.0,
    n_neighbors=n_neighbors if affinity_type == 'nearest_neighbors' else 10,
    random_state=42
)

labels_spec = spec.fit_predict(X)
silhouette_spec = silhouette_score(X, labels_spec)

print(f"📊 Score de Silueta (Spectral): {silhouette_spec:.3f}")

# ==========================
# BLOQUE 6: VISUALIZACIÓN COMPARATIVA
# ==========================
plt.figure(figsize=(15, 5))

# Dataset original
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=30, alpha=0.7, c='navy')
plt.title('DATASET ORIGINAL')
plt.xlabel('Característica X1')
plt.ylabel('Característica X2')
plt.grid(True, alpha=0.3)

# K-means
plt.subplot(1, 3, 2)
colors = ['red', 'blue', 'green', 'orange']
for i in range(k):
    cluster_points = X[labels_km == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                s=30, alpha=0.7, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            marker='X', s=200, c='black', linewidths=2, label='Centros')
plt.title(f'K-MEANS\nSilueta: {silhouette_km:.3f}')
plt.xlabel('Característica X1')
plt.ylabel('Característica X2')
plt.legend()
plt.grid(True, alpha=0.3)

# Clustering Espectral
plt.subplot(1, 3, 3)
for i in range(k):
    cluster_points = X[labels_spec == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                s=30, alpha=0.7, c=colors[i], label=f'Cluster {i+1}')

plt.title(f'CLUSTERING ESPECTRAL\nSilueta: {silhouette_spec:.3f}')
plt.xlabel('Característica X1')
plt.ylabel('Característica X2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================
# BLOQUE 7: ANÁLISIS DE RESULTADOS
# ==========================
print("\n📈 COMPARACIÓN FINAL:")
print("="*45)
print(f"K-means:              {silhouette_km:.3f}")
print(f"Clustering Espectral: {silhouette_spec:.3f}")
print("="*45)

if silhouette_spec > silhouette_km:
    improvement = ((silhouette_spec - silhouette_km) / abs(silhouette_km)) * 100
    print(f" El clustering espectral mejoró en {improvement:.1f}%")
    print("   ¡Capturó mejor la estructura no lineal de los datos!")
else:
    print(" K-means funcionó igual o mejor")
    print("   Prueba ajustando los parámetros del clustering espectral")

