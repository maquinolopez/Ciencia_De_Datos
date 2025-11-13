# =============================================================================
# PRÁCTICA COMPUTACIONAL: AGRUPAMIENTO JERÁRQUICO
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd

print("=== PRÁCTICA: AGRUPAMIENTO JERÁRQUICO ===\n")

# =============================================================================
# GENERAR DATOS DE PRUEBA
# =============================================================================

print("1. GENERANDO DATOS DE PRUEBA...")

# Semilla para reproducibilidad
np.random.seed(42)

# Opción A: Datos esféricos (fáciles para k-means)
X_esferico, y_verdadero_esferico = make_blobs(n_samples=150, centers=3, 
                                             cluster_std=0.8, random_state=42)

# Opción B: Datos no convexos (difíciles para k-means)
X_anillos, y_verdadero_anillos = make_circles(n_samples=150, noise=0.05, 
                                             factor=0.5, random_state=42)

# Opción C: Datos elongados
X_elongado = np.zeros((150, 2))
X_elongado[:50] = np.random.normal(0, 0.3, (50, 2)) + [0, 2]
X_elongado[50:100] = np.random.normal(0, 0.3, (50, 2)) + [2, 0]
X_elongado[100:] = np.random.normal(0, 0.3, (50, 2)) + [4, 2]
y_verdadero_elongado = np.array([0]*50 + [1]*50 + [2]*50)

print("✓ Datos generados:")
print(f"  - Esféricos: {X_esferico.shape}")
print(f"  - Anillos: {X_anillos.shape}")
print(f"  - Elongados: {X_elongado.shape}")

# =============================================================================
# VISUALIZAR DATOS ORIGINALES
# =============================================================================

print("\n2. VISUALIZANDO DATOS ORIGINALES...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Datos esféricos
axes[0].scatter(X_esferico[:, 0], X_esferico[:, 1], c=y_verdadero_esferico, 
               cmap='viridis', s=50, alpha=0.7)
axes[0].set_title('Datos Esféricos\n(Ideales para k-means)')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')

# Datos en anillos
axes[1].scatter(X_anillos[:, 0], X_anillos[:, 1], c=y_verdadero_anillos, 
               cmap='viridis', s=50, alpha=0.7)
axes[1].set_title('Datos No Convexos')
axes[1].set_xlabel('X1')
axes[1].set_ylabel('X2')

# Datos elongados
axes[2].scatter(X_elongado[:, 0], X_elongado[:, 1], c=y_verdadero_elongado, 
               cmap='viridis', s=50, alpha=0.7)
axes[2].set_title('Datos Elongados')
axes[2].set_xlabel('X1')
axes[2].set_ylabel('X2')

plt.tight_layout()
plt.show()


# =============================================================================
# Analisis
# =============================================================================

print("=== SELECCIÓN DE NÚMERO DE CLUSTERS ===\n")

# Usar datos esféricos
X = X_anillos
y_verdadero = y_verdadero_anillos

# Estandarizar datos
scaler = StandardScaler()
X_estandarizado = scaler.fit_transform(X)

# =============================================================================
# PASO 1: APLICAR AGRUPAMIENTO JERÁRQUICO CON SCIKIT-LEARN
# =============================================================================

print("1. APLICANDO AGRUPAMIENTO JERÁRQUICO...")

# Calcular agrupamiento jerárquico completo
Z = linkage(X, method='single')

print("✓ Agrupamiento jerárquico calculado")
print(f"  - Método: Ward")
print(f"  - Número de fusiones: {len(Z)}")

# Visualizar el dendrograma completo
plt.figure(figsize=(10, 6))
dendrogram(Z, orientation='top', truncate_mode='lastp', p=15, show_leaf_counts=True)
plt.title('Dendrograma - Agrupamiento Jerárquico')
plt.xlabel('Puntos o Clusters')
plt.ylabel('Distancia')
plt.show()

# Mostrar información de las primeras fusiones
print("\nPrimeras 5 fusiones:")
print("[Cluster_i, Cluster_j, Distancia, Número_elementos]")
for i in range(min(5, len(Z))):
    print(f"  {i+1:2d}. Clusters {int(Z[i,0]):3d} + {int(Z[i,1]):3d} "
          f"→ Dist: {Z[i,2]:6.3f}, Tamaño: {int(Z[i,3]):2d}")


# =============================================================================
# PASO 2: MÉTODO DEL CODO
# =============================================================================

print("\n2. MÉTODO DEL CODO...")

# Calcular inercias para diferentes números de clusters
rangos_k = range(1, 10)
inercias = []

for k in rangos_k:
    clusters = fcluster(Z, k, criterion='maxclust')
    # criterion='maxclust': "Corta el dendrograma para que me des exactamente k clusters"
    # criterion='distance': "Corta el dendrograma donde la distancia entre clusters sea mayor a algún d  
    
    # Calcular inercia (suma de cuadrados intra-cluster)
    inercia = 0
    for cluster_id in np.unique(clusters):
        puntos_cluster = X_estandarizado[clusters == cluster_id]
        centroide = np.mean(puntos_cluster, axis=0)
        inercia += np.sum((puntos_cluster - centroide) ** 2)
    inercias.append(inercia)

# Visualizar método del codo
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(rangos_k, inercias, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Suma de Cuadrados Intra-Cluster')
plt.title('Método del Codo')
plt.grid(True, alpha=0.3)

# =============================================================================
# PASO 3: MÉTODO DE LA SILUETA
# =============================================================================

print("3. MÉTODO DE LA SILUETA...")

# Calcular scores de silueta para diferentes k
siluetas = []

for k in rangos_k:
    if k == 1:
        siluetas.append(0)  # Silueta no definida para k=1
    else:
        clusters = fcluster(Z, k, criterion='maxclust')
        silueta = silhouette_score(X_estandarizado, clusters)
        siluetas.append(silueta)

# Encontrar mejor k
mejor_k = rangos_k[np.argmax(siluetas)]
print(f"✓ Mejor k según silueta: {mejor_k}")

# Visualizar método de silueta
plt.subplot(1, 2, 2)
plt.plot(rangos_k, siluetas, 'ro-', linewidth=2, markersize=6)
plt.axvline(x=mejor_k, color='red', linestyle='--', label=f'Mejor k = {mejor_k}')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Score de Silueta')
plt.title('Método de la Silueta')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# PASO 4: RESULTADOS CON EL MEJOR K
# =============================================================================

print(f"\n4. RESULTADOS CON k = {mejor_k}")

# Obtener clusters con el mejor k
clusters_final = fcluster(Z, mejor_k, criterion='maxclust')

# Calcular métricas
silueta_final = silhouette_score(X_estandarizado, clusters_final)
rand_final = adjusted_rand_score(y_verdadero, clusters_final)

print(f"✓ Score de Silueta: {silueta_final:.3f}")
print(f"✓ Adjusted Rand Index: {rand_final:.3f}")

# Mostrar distribución de clusters
unique, counts = np.unique(clusters_final, return_counts=True)
print("✓ Distribución de puntos:")
for cluster_id, count in zip(unique, counts):
    print(f"  - Cluster {cluster_id}: {count} puntos")

# Visualizar resultados
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=clusters_final, cmap='tab10', s=50, alpha=0.7)
plt.title(f'Clustering Jerárquico\nk = {mejor_k}, Silueta = {silueta_final:.3f}')
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_verdadero, cmap='tab10', s=50, alpha=0.7)
plt.title('Verdadero')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()


# =============================================================================
# COMPARAR DIFERENTES MÉTODOS DE ENLACE
# =============================================================================

print("\n3. COMPARANDO MÉTODOS DE ENLACE...")

# Usaremos los datos elongados para mostrar las diferencias
X = X_anillos
y_verdadero = y_verdadero_anillos



# Estandarizar los datos (importante para Ward)
scaler = StandardScaler()
X_estandarizado = scaler.fit_transform(X)

# Métodos de enlace a comparar
metodos = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, metodo in enumerate(metodos):
    print(f"  Probando método: {metodo}")
    
    # Calcular agrupamiento jerárquico
    Z = linkage(X_estandarizado, method=metodo)
        
    # Dendrograma
    plt.sca(axes[0, i])
    dendrogram(Z, orientation='top', truncate_mode='lastp', p=12)
    plt.title(f'Dendrograma - {metodo.upper()}')
    plt.xlabel('Puntos')
    plt.ylabel('Distancia')
    
   
    # Calcular correlación cofenética
    coph_corr, coph_dist = cophenet(Z, pdist(X_estandarizado))
    print(f"    Correlación cofenética: {coph_corr:.3f}")
    
    
    # Cortar dendrograma para obtener 3 clusters
    clusters = fcluster(Z, 3, criterion='maxclust')
    
    # Calcular silueta
    silueta = silhouette_score(X_estandarizado, clusters)
    print(f"    Score de silueta: {silueta:.3f}")
    
    # Calcular adjusted Rand index (comparación con verdadero)
    rand_index = adjusted_rand_score(y_verdadero, clusters)
    print(f"    Adjusted Rand Index: {rand_index:.3f}")
    

    plt.sca(axes[1, i])
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='tab10', s=50, alpha=0.7)
    plt.title(f'Clusters - {metodo.upper()}\nSilueta: {silueta:.3f}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# =============================================================================
# ANÁLISIS DE SELECCIÓN DE NÚMERO DE CLUSTERS
# =============================================================================

# Usar el mejor método según evaluación anterior
mejor_metodo = 'ward'
Z = linkage(X_estandarizado, method=mejor_metodo)

# Probar diferentes números de clusters
rangos_k = range(2, 8)
siluetas = []
inertias = []  # Suma de cuadrados intra-cluster

for k in rangos_k:
    # Obtener clusters
    clusters = fcluster(Z, k, criterion='maxclust')
    
    # Calcular métricas
    silueta = silhouette_score(X_estandarizado, clusters)
    siluetas.append(silueta)
    
    # Calcular inercia (suma de cuadrados intra-cluster)
    inercia = 0
    for cluster_id in np.unique(clusters):
        puntos_cluster = X_estandarizado[clusters == cluster_id]
        centroide = np.mean(puntos_cluster, axis=0)
        inercia += np.sum((puntos_cluster - centroide) ** 2)
    inertias.append(inercia)

# Visualizar métricas
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico de silueta
axes[0].plot(rangos_k, siluetas, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Score de Silueta')
axes[0].set_title('Método de la Silueta\n(Maximizar)')
axes[0].grid(True, alpha=0.3)

# Gráfico de codo
axes[1].plot(rangos_k, inertias, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Suma de Cuadrados Intra-Cluster')
axes[1].set_title('Método del Codo\n(Punto de inflexión)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Encontrar el mejor k según silueta
mejor_k = rangos_k[np.argmax(siluetas)]
print(f"✓ Mejor número de clusters según silueta: k = {mejor_k}")

# =============================================================================
# COMPARACIÓN k-means vs JERÁRQUICO Revisar que datos estamos usando
# =============================================================================
from sklearn.cluster import KMeans

# Usar datos esféricos para la comparación
X_comp = X_anillos
y_verdadero_comp = y_verdadero_anillos

# Estandarizar
X_comp_estandarizado = scaler.fit_transform(X_comp)

# Crear figura con 2 filas y 3 columnas
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# =============================================================================
# k-MEANS
# =============================================================================

kmeans = KMeans(n_clusters=3, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_comp_estandarizado)
centroides = kmeans.cluster_centers_

axes[0, 0].scatter(X_comp[:, 0], X_comp[:, 1], c=clusters_kmeans, cmap='tab10', s=50, alpha=0.7)
axes[0, 0].set_title('k-means')
axes[0, 0].set_xlabel('X1')
axes[0, 0].set_ylabel('X2')

# =============================================================================
# JERÁRQUICO - SINGLE LINKAGE
# =============================================================================

Z_single = linkage(X_comp_estandarizado, method='single')
clusters_single = fcluster(Z_single, 3, criterion='maxclust')

axes[0, 1].scatter(X_comp[:, 0], X_comp[:, 1], c=clusters_single, cmap='tab10', s=50, alpha=0.7)
axes[0, 1].set_title('Jerárquico - Single')
axes[0, 1].set_xlabel('X1')
axes[0, 1].set_ylabel('X2')

# =============================================================================
# JERÁRQUICO - COMPLETE LINKAGE
# =============================================================================

Z_complete = linkage(X_comp_estandarizado, method='complete')
clusters_complete = fcluster(Z_complete, 3, criterion='maxclust')

axes[0, 2].scatter(X_comp[:, 0], X_comp[:, 1], c=clusters_complete, cmap='tab10', s=50, alpha=0.7)
axes[0, 2].set_title('Jerárquico - Complete')
axes[0, 2].set_xlabel('X1')
axes[0, 2].set_ylabel('X2')

# =============================================================================
# JERÁRQUICO - AVERAGE LINKAGE
# =============================================================================

Z_average = linkage(X_comp_estandarizado, method='average')
clusters_average = fcluster(Z_average, 3, criterion='maxclust')

axes[1, 0].scatter(X_comp[:, 0], X_comp[:, 1], c=clusters_average, cmap='tab10', s=50, alpha=0.7)
axes[1, 0].set_title('Jerárquico - Average')
axes[1, 0].set_xlabel('X1')
axes[1, 0].set_ylabel('X2')

# =============================================================================
# JERÁRQUICO - WARD
# =============================================================================

Z_ward = linkage(X_comp_estandarizado, method='ward')
clusters_ward = fcluster(Z_ward, 3, criterion='maxclust')

axes[1, 1].scatter(X_comp[:, 0], X_comp[:, 1], c=clusters_ward, cmap='tab10', s=50, alpha=0.7)
axes[1, 1].set_title('Jerárquico - Ward')
axes[1, 1].set_xlabel('X1')
axes[1, 1].set_ylabel('X2')

# =============================================================================
# VERDADERO (para comparación)
# =============================================================================

axes[1, 2].scatter(X_comp[:, 0], X_comp[:, 1], c=y_verdadero_comp, cmap='tab10', s=50, alpha=0.7)
axes[1, 2].set_title('Verdadero')
axes[1, 2].set_xlabel('X1')
axes[1, 2].set_ylabel('X2')

plt.tight_layout()
plt.show()