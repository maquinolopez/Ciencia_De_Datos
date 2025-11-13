# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris, load_wine
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# Generar datos simulados correlacionados
np.random.seed(42)
n_samples = 300

# Crear variables correlacionadas
var1 = np.random.normal(-1, 1, n_samples)
var2 = 0.7 * var1 + np.random.normal(0, 0.3, n_samples)
var3 = 0.5 * var1 + 0.5 * var2 + np.random.normal(0, 0.2, n_samples)
var4 = np.random.normal(5, .5, n_samples)  # Variable independiente

# Crear DataFrame
data_simulado = pd.DataFrame({
    'Variable_1': var1,
    'Variable_2': var2,
    'Variable_3': var3,
    'Variable_4': var4
})

print("Datos Simulados - Primeras filas:")
print(data_simulado.head())
print(f"\nMatriz de correlación:")
print(data_simulado.corr().round(3))

# Estandarizar los datos
scaler = StandardScaler()
data_estandarizado = scaler.fit_transform(data_simulado)

# Aplicar PCA
pca = PCA()
componentes_principales = pca.fit_transform(data_estandarizado)

# Crear DataFrame con componentes principales
df_pca = pd.DataFrame(
    data=componentes_principales,
    columns=[f'PC{i+1}' for i in range(componentes_principales.shape[1])]
)

# Análisis de varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

print(f"\nVarianza explicada por cada componente: {varianza_explicada}")
print(f"Varianza acumulada: {varianza_acumulada}")

# Visualización
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico de varianza explicada
axes[0,0].bar(range(1, len(varianza_explicada)+1), varianza_explicada, alpha=0.6)
axes[0,0].plot(range(1, len(varianza_acumulada)+1), varianza_acumulada, 'ro-')
axes[0,0].set_xlabel('Componente Principal')
axes[0,0].set_ylabel('Varianza Explicada')
axes[0,0].set_title('Varianza Explicada por Componentes Principales')
axes[0,0].grid(True)

# Scores de los componentes
axes[0,1].scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7)
axes[0,1].set_xlabel(f'PC1 ({varianza_explicada[0]:.2%} varianza)')
axes[0,1].set_ylabel(f'PC2 ({varianza_explicada[1]:.2%} varianza)')
axes[0,1].set_title('Proyección en PC1 vs PC2')

# Loadings (coeficientes de las variables originales en los componentes)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

axes[1,0].barh(data_simulado.columns, loadings[:, 0])
axes[1,0].set_title('Loadings - PC1')
axes[1,0].set_xlabel('Contribución')

axes[1,1].barh(data_simulado.columns, loadings[:, 1])
axes[1,1].set_title('Loadings - PC2')
axes[1,1].set_xlabel('Contribución')

plt.tight_layout()
plt.show()

# Generar datos para clustering
X, y_true = make_blobs(n_samples=400, centers=4, n_features=6, 
                       cluster_std=1.5, random_state=42)

print(f"Forma de los datos originales: {X.shape}")

# Aplicar PCA para reducción dimensional
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

print(f"Varianza explicada por los 2 componentes: {pca.explained_variance_ratio_.sum():.3f}")

# Aplicar K-Means en el espacio reducido por PCA
kmeans_pca = KMeans(n_clusters=4, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca)

# Comparar con K-Means en datos originales
kmeans_original = KMeans(n_clusters=4, random_state=42)
clusters_original = kmeans_original.fit_predict(StandardScaler().fit_transform(X))

# Calcular métricas de evaluación
silhouette_original = silhouette_score(StandardScaler().fit_transform(X), clusters_original)
silhouette_pca = silhouette_score(X_pca, clusters_pca)

print(f"Silhouette Score - Datos originales: {silhouette_original:.3f}")
print(f"Silhouette Score - PCA + K-Means: {silhouette_pca:.3f}")

# Visualización comparativa
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Datos originales (primeras dos dimensiones)
scatter0 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Datos Originales (Verdaderos Clusters)')
plt.colorbar(scatter0, ax=axes[0])

# K-Means en datos originales
scatter1 = axes[1].scatter(X[:, 0], X[:, 1], c=clusters_original, cmap='viridis', alpha=0.7)
axes[1].scatter(kmeans_original.cluster_centers_[:, 0], 
                kmeans_original.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='red')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('K-Means en Datos Originales')
plt.colorbar(scatter1, ax=axes[1])

# K-Means después de PCA
scatter2 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_pca, cmap='viridis', alpha=0.7)
axes[2].scatter(kmeans_pca.cluster_centers_[:, 0], 
                kmeans_pca.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='red')
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[2].set_title('PCA + K-Means')
plt.colorbar(scatter2, ax=axes[2])

plt.tight_layout()
plt.show()

# Análisis de la relación entre componentes y clusters originales
df_analysis = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster_PCA': clusters_pca,
    'Cluster_Original': y_true
})

print("\nAnálisis de clusters por componente principal:")
print(df_analysis.groupby('Cluster_PCA')[['PC1', 'PC2']].mean())






















