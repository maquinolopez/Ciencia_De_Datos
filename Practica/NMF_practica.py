# -*- coding: utf-8 -*-
"""
NMF (Non‑negative Matrix Factorization) — versión didáctica *inline*
-------------------------------------------------------------------
Idea: trabajar en Spyder ejecutando el script de arriba a abajo, sin
encapsular en funciones. Cada bloque muestra:
  1) qué hace, 2) cómo se usa en scikit‑learn, 3) qué salida esperar.
Reglas: sólo matplotlib (sin seaborn); una gráfica por figura.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

np.random.seed(42)
plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "grid.alpha": 0.3})

# =============================================================================
# 1) CARGA Y EXPLORACIÓN BÁSICA (sin funciones)
# =============================================================================
print("\n[1] Carga del dataset de dígitos")
digits = load_digits()
X = digits.data        # (n, 64) — imágenes 8x8 vectorizadas
y = digits.target      # etiquetas 0..9
print("Forma X:", X.shape)
print("Clases:", np.unique(y))
print("Rango de píxeles (0..16):", float(X.min()), float(X.max()))

# Muestra rápida de ejemplos
print("Mostrando 10 ejemplos...")
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = np.ravel(axes)
for i in range(10):
    axes[i].imshow(X[i].reshape(8, 8), cmap="gray")
    axes[i].set_title(f"y={y[i]}", fontsize=9)
    axes[i].axis("off")
fig.suptitle("Ejemplos del dataset")
fig.tight_layout()
plt.show()

# =============================================================================
# 2) PREPROCESADO MÍNIMO PARA NMF (no negatividad y escala comparable)
# =============================================================================
print("\n[2] Escalado a [0,1] con MinMaxScaler (recomendado para NMF)")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Comprobación: rango X_scaled -> ", float(X_scaled.min()), float(X_scaled.max()))

# =============================================================================
# 3) NMF: AJUSTE Y MEDIDA DE CALIDAD (error de reconstrucción)
# =============================================================================
print("\n[3] Ajuste NMF para distintos k y cálculo de error de reconstrucción")
ks = [5, 10, 20, 30, 40]
errors = []
for k in ks:
    nmf_k = NMF(n_components=k, init="random", max_iter=500, random_state=42)
    W_k = nmf_k.fit_transform(X_scaled)    # (n, k) — mezclas no negativas
    H_k = nmf_k.components_               # (k, 64) — "partes" no negativas
    err_k = nmf_k.reconstruction_err_      # ||X - W@H||_F
    errors.append(err_k)
    print(f"  k={k:2d} | error={err_k:.4f}")

plt.figure()
plt.plot(ks, errors, marker="o")
plt.xlabel("Número de componentes (k)")
plt.ylabel("Error de reconstrucción")
plt.title("NMF — Error vs k")
plt.show()

# =============================================================================
# 4) ¿QUÉ SON W Y H? VISUALIZAR COMPONENTES Y UNA RECONSTRUCCIÓN CONCRETA
# =============================================================================
print("\n[4] Visualización de componentes (H) y reconstrucción de un ejemplo")
k = 10
nmf = NMF(n_components=k, init="random", max_iter=500, random_state=42)
W = nmf.fit_transform(X_scaled)
H = nmf.components_

# Mostrar hasta 10 componentes (H)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = np.ravel(axes)
for i in range(10):
    axes[i].imshow(H[i].reshape(8, 8), cmap="viridis")
    axes[i].set_title(f"Comp {i}", fontsize=9)
    axes[i].axis("off")
fig.suptitle("Componentes NMF (H)")
fig.tight_layout()
plt.show()

# Reconstrucción de un ejemplo puntual
idx = 0  # cambiar en vivo para ver otros dígitos
recon = W[idx] @ H
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(X[idx].reshape(8, 8), cmap="gray"); ax[0].set_title(f"Original\ny={y[idx]}")
ax[0].axis("off")
ax[1].imshow(recon.reshape(8, 8), cmap="gray"); ax[1].set_title("Reconstruida")
ax[1].axis("off")
diff = np.abs(X[idx].reshape(8, 8) - recon.reshape(8, 8))
im = ax[2].imshow(diff, cmap="hot"); ax[2].set_title("|X - WH|")
ax[2].axis("off"); fig.colorbar(im, ax=ax[2], shrink=0.8)
fig.tight_layout(); 
plt.show()

# Pesos por componente para ese ejemplo
plt.figure()
plt.plot(np.arange(k), W[idx], marker="o")
plt.xlabel("Componente NMF"); plt.ylabel("Peso W[idx, :]")
plt.title(f"Contribución por componente (ejemplo {idx})")
plt.show()

# =============================================================================
# 5) COMPARACIÓN BREVE CON PCA: VARIANZA EXPLICADA vs ERROR NMF
# =============================================================================
print("\n[5] PCA (varianza explicada) vs NMF (error)")
k_pca = 10
pca = PCA(n_components=k_pca, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(np.arange(1, k_pca+1), var_exp, marker="o")
plt.xlabel("# componentes (PCA)"); plt.ylabel("Varianza explicada acumulada")
plt.title("PCA — Varianza explicada")
plt.show()

print(f"  Varianza explicada (k={k_pca}): {var_exp[-1]:.4f}")

# Para comparar números, volvemos a leer un error NMF (ya calculado para k=10)
err_nmf_k10 = errors[ks.index(10)] if 10 in ks else np.nan
print(f"  Error NMF (k=10): {err_nmf_k10:.4f}")

# =============================================================================
# 6) CLASIFICACIÓN SIMPLE: LOGISTIC REGRESSION sobre representaciones NMF y PCA
# =============================================================================
print("\n[6] Clasificación con Regresión Logística tras NMF/PCA")
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
ks_cls = [5, 10, 20, 40]
acc_nmf, acc_pca = [], []

for k in ks_cls:
    # NMF -> RL
    nmf_k = NMF(n_components=k, init="random", max_iter=500, random_state=42)
    Xtr_nmf = nmf_k.fit_transform(Xtr)
    Xte_nmf = nmf_k.transform(Xte)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr_nmf, ytr)
    acc_nmf.append(accuracy_score(yte, clf.predict(Xte_nmf)))

    # PCA -> RL
    pca_k = PCA(n_components=k, random_state=42)
    Xtr_pca = pca_k.fit_transform(Xtr)
    Xte_pca = pca_k.transform(Xte)
    clf.fit(Xtr_pca, ytr)
    acc_pca.append(accuracy_score(yte, clf.predict(Xte_pca)))

print(" k  |  acc(NMF)  acc(PCA)")
print("----+---------------------")
for k, a_n, a_p in zip(ks_cls, acc_nmf, acc_pca):
    print(f"{k:>3} |  {a_n:8.4f}   {a_p:8.4f}")

plt.figure(); plt.plot(ks_cls, acc_nmf, marker="o")
plt.xlabel("k"); plt.ylabel("Accuracy test"); plt.title("Accuracy — NMF")
plt.show()

plt.figure(); plt.plot(ks_cls, acc_pca, marker="o")
plt.xlabel("k"); plt.ylabel("Accuracy test"); plt.title("Accuracy — PCA")
plt.show()

# Reporte del mejor NMF
best_idx = int(np.argmax(acc_nmf))
best_k = ks_cls[best_idx]
print(f"\nMejor NMF: k={best_k} con accuracy={acc_nmf[best_idx]:.4f}")
nmf_best = NMF(n_components=best_k, init="random", max_iter=500, random_state=42)
Xtr_nmf = nmf_best.fit_transform(Xtr)
Xte_nmf = nmf_best.transform(Xte)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Xtr_nmf, ytr)
ypred = clf.predict(Xte_nmf)
print("\nReporte de clasificación (mejor NMF):\n")
print(classification_report(yte, ypred))

# =============================================================================
# 7) PROYECCIONES 2D (NMF vs PCA) Y SEPARABILIDAD (Silhouette)
# =============================================================================
print("\n[7] Proyecciones 2D y silhouette score")
nmf2 = NMF(n_components=2, init="random", max_iter=500, random_state=42)
X2_nmf = nmf2.fit_transform(X_scaled)

pca2 = PCA(n_components=2, random_state=42)
X2_pca = pca2.fit_transform(X_scaled)

plt.figure(); sc = plt.scatter(X2_nmf[:, 0], X2_nmf[:, 1], c=y, s=12, alpha=0.7)
plt.xlabel("Comp 1"); plt.ylabel("Comp 2"); plt.title("NMF 2D"); plt.colorbar(sc, label="dígito")
plt.show()

plt.figure(); sc = plt.scatter(X2_pca[:, 0], X2_pca[:, 1], c=y, s=12, alpha=0.7)
plt.xlabel("PC 1"); plt.ylabel("PC 2"); plt.title("PCA 2D"); plt.colorbar(sc, label="dígito")
plt.show()

sil_nmf = silhouette_score(X2_nmf, y)
sil_pca = silhouette_score(X2_pca, y)
print(f"  Silhouette NMF 2D: {sil_nmf:.4f}")
print(f"  Silhouette PCA 2D: {sil_pca:.4f}")

# =============================================================================
# 8) PREGUNTAS RÁPIDAS DE DISCUSIÓN
# =============================================================================
print("\n[8] Preguntas de reflexión")
print(" - ¿Qué interpretas en H (partes) y en W (mezclas)?")
print(" - ¿Cómo elegir k según la tarea (reconstrucción vs clasificación)?")
print(" - ¿Qué ventajas/desventajas observaste vs PCA?")
print(" - ¿Cómo afecta la inicialización aleatoria y max_iter?")
