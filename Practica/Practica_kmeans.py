"""
Práctica guiada: K-means con scikit-learn
Requisitos mínimos
------------------
pip install numpy pandas matplotlib scikit-learn
"""

# =============================================================================
# 0. IMPORTS Y CONFIGURACIÓN
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# Configuración gráfica mínima
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["axes.grid"] = True
np.set_printoptions(precision=3, suppress=True)

RANDOM_STATE = 42  # Cambia y observa cómo varía el resultado
np.random.seed(RANDOM_STATE)


# =============================================================================
# 1. GENERAR DATOS SINTÉTICOS (SIN ETIQUETAS)
# =============================================================================

# make_blobs crea nubes de puntos alrededor de centros "reales" que NO usaremos
# como etiquetas. Solo se usan para verificar que k-means recupera estructuras.
n_samples = 1000
n_features = 2     # para visualizar fácilmente; puedes subirlo y usar PCA
true_centers = 5
cluster_std = 0.9  # mayor std = clusters más solapados (más difícil)

X, y_true = make_blobs(n_samples=n_samples,
                       centers=true_centers,
                       cluster_std=cluster_std,
                       n_features=n_features,
                       random_state=RANDOM_STATE)

print("Dimensión de X:", X.shape)

# Visualización básica (sin etiquetas)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=20)
plt.title("Datos sin etiquetas (solo para inspección visual)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.show()


# =============================================================================
# 2. ESCALADO / ESTANDARIZACIÓN
# =============================================================================

# k-means usa distancias Euclidianas; escalar ayuda cuando hay variables
# en unidades diferentes o con distinta variabilidad.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 3. EXPLORAR K (MÉTODO DEL CODO Y SILHOUETTE)
# =============================================================================

# Rango de K a evaluar (evita K=1 para Silhouette)
K_range = range(2, 11)

SSE = []            # Inertia en scikit-learn = suma de distancias cuadradas intra-cluster
silhouette_avg = [] # Promedio global de Silhouette por K

for k in K_range:
    km = KMeans(n_clusters=k,
                init="k-means++",  # inicialización recomendada
                n_init=10,         # reinicios para evitar mínimos locales
                max_iter=300,
                random_state=RANDOM_STATE)
    km.fit(X_scaled)
    SSE.append(km.inertia_)
    labels = km.labels_
    sil = silhouette_score(X_scaled, labels)
    silhouette_avg.append(sil)
    print(f"K={k:2d} | SSE={km.inertia_:10.2f} | Silhouette={sil: .4f}")

# ---- Gráfica del codo (SSE) ----
plt.figure()
plt.plot(list(K_range), SSE, marker="o", linewidth=2)
plt.title("Método del codo (SSE vs K)")
plt.xlabel("Número de clusters K")
plt.ylabel("SSE (Inertia)")
plt.tight_layout()
plt.show()

# ---- Gráfica de Silhouette promedio ----
plt.figure()
plt.plot(list(K_range), silhouette_avg, marker="o", linewidth=2)
plt.title("Silhouette promedio vs K")
plt.xlabel("Número de clusters K")
plt.ylabel("Silhouette promedio")
plt.tight_layout()
plt.show()

# Sugerencia automática simple de K: el que maximiza Silhouette (heurística)
k_sug = int(list(K_range)[int(np.argmax(silhouette_avg))])
print(f"\nHeurística rápida → K sugerido (max Silhouette): {k_sug}\n")


# =============================================================================
# 4. ENTRENAR K-MEANS CON K OPTIMO (AJUSTA K MANUALMENTE SI QUIERES)
# =============================================================================

K_OPT = k_sug  # Puedes forzar otro valor manualmente, p. ej., K_OPT = 4

kmeans = KMeans(n_clusters=K_OPT,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=RANDOM_STATE)

labels_opt = kmeans.fit_predict(X_scaled)
centers_scaled = kmeans.cluster_centers_

print(f"Centroides (en espacio escalado) para K={K_OPT}:\n", centers_scaled)
sil_opt = silhouette_score(X_scaled, labels_opt)
print(f"Silhouette promedio para K={K_OPT}: {sil_opt:.4f}")

# ---- Visualización en 2D (ya estamos en 2D) ----
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_opt, s=20)
plt.scatter(centers_scaled[:, 0], centers_scaled[:, 1], marker="X", s=200)
plt.title(f"K-means con K={K_OPT} (espacio escalado)")
plt.xlabel("x1 (escalado)")
plt.ylabel("x2 (escalado)")
plt.tight_layout()
plt.show()



# =============================================================================
# 5. REINICIOS / SEMILLAS DISTINTAS (ROBUSTEZ) - OPCIONAL
# =============================================================================

# Demostración rápida: cómo cambian SSE y Silhouette con distintas semillas
seeds = [1000, 7611, 1213, 4542, 1123,1345]
resumen = []

for seed in seeds:
    km = KMeans(n_clusters=K_OPT, init="k-means++", n_init=10, random_state=seed)
    labels_seed = km.fit_predict(X_scaled)
    sse_seed = km.inertia_
    sil_seed = silhouette_score(X_scaled, labels_seed)
    resumen.append({"seed": seed, "SSE": sse_seed, "Silhouette": sil_seed})

df_resumen = pd.DataFrame(resumen).sort_values("Silhouette", ascending=False)
print("\nVariación por semilla (misma K):\n", df_resumen.to_string(index=False))

plt.figure()
plt.plot(df_resumen["seed"], df_resumen["Silhouette"], marker="o", linewidth=2)
plt.title(f"Silhouette vs. semilla (K={K_OPT})")
plt.xlabel("Semilla (random_state)")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.show()

