# ============================================
# Práctica: Clustering con EM (GaussianMixture, scikit-learn)
# --------------------------------------------
# Objetivos:
# 1) Generar datos elípticos (anisotrópicos) para evidenciar ventajas de EM.
# 2) Seleccionar K con BIC/AIC.
# 3) Ajustar GMM con EM y visualizar pertenencias suaves + elipses.
# 4) Comparar vs KMeans (ARI y Silhouette).
# ============================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# -------------------------
# 0) Semilla y estilo base
# -------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# -------------------------
# 1) Datos sintéticos
#    - 3 centros, luego aplicamos una transformación lineal
#      para hacer los clusters elípticos/rotados.
# -------------------------
n_samples = 500
X, y_true = make_blobs(n_samples=n_samples,
                       centers=[(-3, 3), (0, 0), (4, 2)],
                       cluster_std=[0.8, 1.2, 0.9],
                       random_state=RANDOM_SEED)

# Transformación para crear elipses (mezcla de rotación + escala)
A = np.array([[1.8, 1.2],
              [0.4, 1.4]])
X = X @ A.T

# Escalamos (suele ayudar numéricamente, no obligatorio)
X = StandardScaler().fit_transform(X)

# Visualización inicial
plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7,c="blue")
plt.title("Datos (elípticos) para clustering")
plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 2) Función utilitaria: dibujar elipses
# ---------------------------------------
from matplotlib.patches import Ellipse

def draw_cov_ellipse(ax, mean, cov, nsig=2.0, **kwargs):
    """Dibuja una elipse (nsig ~ número de desviaciones) a partir de la covarianza."""
    vals, vecs = np.linalg.eigh(cov)            # autovalores/auto-vectores
    order = vals.argsort()[::-1]                # ordenar mayor->menor
    vals, vecs = vals[order], vecs[:, order]
    # Ángulo de la elipse en grados (componente principal):
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # Diámetros (2*nsig*sqrt(lambda))
    width, height = 2*nsig*np.sqrt(vals[0]), 2*nsig*np.sqrt(vals[1])
    e = Ellipse(mean, width, height, angle=angle, fill=False, lw=2, **kwargs)
    ax.add_patch(e)
    return e

# -------------------------------------------------------
# 3) Selección de K con BIC/AIC para GaussianMixture (EM)
# -------------------------------------------------------
Ks = range(1, 8)                # probamos K=1..7
covariance_type = "full"        # "full", "tied", "diag", "spherical"
bic_list, aic_list, models = [], [], []

for K in Ks:
    # n_init controla cuántas inicializaciones por K (para evitar malos mínimos locales)
    gmm = GaussianMixture(n_components=K,
                          covariance_type=covariance_type,
                          init_params="kmeans",
                          n_init=5,
                          random_state=RANDOM_SEED)
    gmm.fit(X)
    bic_list.append(gmm.bic(X))
    aic_list.append(gmm.aic(X))
    models.append(gmm)

# Elegimos mejor K por BIC (menor es mejor)
best_idx = int(np.argmin(bic_list))
best_K = Ks[best_idx]
best_gmm = models[best_idx]

print(f"Mejor K por BIC: {best_K}")
print(f"BIC(K={best_K}) = {bic_list[best_idx]:.1f}, AIC = {aic_list[best_idx]:.1f}")

# Graficamos BIC/AIC
plt.figure(figsize=(6, 4))
plt.plot(Ks, bic_list, marker="o", label="BIC")
plt.plot(Ks, aic_list, marker="s", label="AIC")
plt.axvline(best_K, ls="--", alpha=0.6)
plt.title(f"Selección de K (covariance_type='{covariance_type}')")
plt.xlabel("K"); plt.ylabel("Criterio (menor mejor)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 4) Ajuste final con el mejor K y extracción de resultados
#    - Responsibilities (P(z=k|x)) con predict_proba
#    - Etiquetas duras con predict
# -------------------------------------------------------
gmm = GaussianMixture(n_components=best_K,
                      covariance_type=covariance_type,
                      init_params="kmeans",
                      n_init=5,
                      random_state=RANDOM_SEED)
gmm.fit(X)

probs = gmm.predict_proba(X)       # responsabilidades γ_{ik}
labels_em = gmm.predict(X)         # etiquetas duras (argmax_k γ_{ik})
means = gmm.means_
covs = gmm.covariances_

# -------------------------------------------------------
# 5) Visualización: pertenencias suaves + elipses 2σ
#    - alpha y tamaño ~ max_k γ_{ik}
#    - elipses centradas en medias con covarianzas del GMM
# -------------------------------------------------------
max_resp = probs.max(axis=1)             # intensidad de pertenencia
markers = ["o", "s", "^", "D", "P", "X"]

plt.figure(figsize=(6, 5))
for k in range(best_K):
    mask = labels_em == k
    if np.any(mask):
        plt.scatter(X[mask, 0], X[mask, 1],
                    s=18 + 40*max_resp[mask],
                    alpha=np.clip(max_resp[mask], 0.25, 1.0),
                    marker=markers[k % len(markers)],
                    edgecolors="none",
                    label=f"Cluster {k}")

# Medias y elipses (2σ)
for k in range(best_K):
    plt.plot(means[k, 0], means[k, 1], marker="x", ms=9)
    draw_cov_ellipse(plt.gca(), means[k], covs[k], nsig=2.0)

plt.title(f"GMM-EM (K={best_K}, cov='{covariance_type}')")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(loc="best", framealpha=0.8)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 6) Comparación vs KMeans
#    - Ajustamos KMeans con mismo K
#    - Métricas: ARI (si tenemos y_true) y Silhouette
# -------------------------------------------------------
kmeans = KMeans(n_clusters=best_K, n_init=10, random_state=RANDOM_SEED)
labels_km = kmeans.fit_predict(X)

# Silhouette (independiente de y_true; para clustering puro)
sil_em = silhouette_score(X, labels_em)
sil_km = silhouette_score(X, labels_km)

print(f"\nSilhouette  GMM-EM : {sil_em:.3f}")
print(f"Silhouette  KMeans : {sil_km:.3f}")

# ARI sólo tiene sentido aquí porque conocemos y_true (datos sintéticos).
# En datos reales, ARI no aplica (no hay etiquetas de verdad).
ari_em = adjusted_rand_score(y_true, labels_em)
ari_km = adjusted_rand_score(y_true, labels_km)
print(f"ARI vs verdad (sintético) - EM: {ari_em:.3f}, KMeans: {ari_km:.3f}")

