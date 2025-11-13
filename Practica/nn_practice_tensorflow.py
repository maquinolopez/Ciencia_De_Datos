#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN Practice with TensorFlow — Versión didáctica y lineal (Spyder-friendly)
===========================================================================

Objetivo:
  Explorar paso a paso cómo construir y entrenar redes neuronales con TensorFlow/Keras
  para clasificación y regresión usando datos simulados y reales, y una CNN sencilla
  para imágenes (MNIST). Todo en un solo archivo, sin funciones "ocultas", para
  poder explicar línea por línea en clase.

Requisitos:
  - tensorflow >= 2.11
  - numpy, matplotlib, scikit-learn

Ejecuta este archivo completo en Spyder (F5). Cada bloque imprime su progreso.
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.datasets import make_moons, load_iris, fetch_california_housing

# -----------------------------------------------------------------------------
# Configuración básica: semilla y (si existe) uso de GPU sin reservar toda la memoria
# -----------------------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[INFO] GPU detectada(s): {len(gpus)}. Activado memory growth.")
    except Exception as e:
        print("[WARN] No fue posible activar memory growth:", e)
else:
    print("[INFO] Ejecutando en CPU (no se detectó GPU).")

print("\nVersiones:")
print("  numpy:", np.__version__)
print("  tensorflow:", tf.__version__)

# =============================================================================
# PARTE 1 — CLASIFICACIÓN (Simulado: make_moons) con MLP
# =============================================================================
print("\n=== PARTE 1 — Clasificación (Simulado: make_moons) ===")

# 1.1 Generamos datos no linealmente separables (dos lunas)
X, y = make_moons(n_samples=1200, noise=0.25, random_state=0)
# 1.2 División entrenamiento / prueba con estratificación
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# 1.3 Estandarización (muy importante para MLP)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# 1.4 Definimos el MLP (capas densas) para clasificación multiclase
model_cls = models.Sequential()
model_cls.add(layers.Input(shape=(Xtr_s.shape[1],)))
model_cls.add(layers.Dense(64, activation="relu"))
model_cls.add(layers.Dropout(0.15))
model_cls.add(layers.Dense(64, activation="relu"))
model_cls.add(layers.Dense(2, activation="softmax"))  # 2 clases

# 1.5 Compilamos: pérdida adecuada + métrica
model_cls.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

# 1.6 Entrenamos con EarlyStopping y ReduceLROnPlateau
cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True) # esto detiene el modelo antes de que empieze a tener sobre ajuste
cb_rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1) # alentar el proceso de aprendisaje 

hist1 = model_cls.fit(Xtr_s, ytr,
                      validation_split=0.2,
                      epochs=80,
                      batch_size=64,
                      callbacks=[cb_early, cb_rlr],
                      verbose=0)

# 1.7 Curvas de aprendizaje
plt.figure(figsize=(7, 4))
plt.plot(hist1.history["loss"], label="loss (train)")
plt.plot(hist1.history["val_loss"], label="loss (val)")
if "accuracy" in hist1.history:
    plt.plot(hist1.history["accuracy"], label="acc (train)")
if "val_accuracy" in hist1.history:
    plt.plot(hist1.history["val_accuracy"], label="acc (val)")
plt.title("Moons — MLP (curvas de aprendizaje)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 1.8 Evaluación
ypred = np.argmax(model_cls.predict(Xte_s, verbose=0), axis=1)
acc = accuracy_score(yte, ypred)
print(f"Accuracy (test): {acc:.3f}")

# 1.9 Frontera de decisión (2D)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_s = scaler.transform(grid)
zz = np.argmax(model_cls.predict(grid_s, verbose=0), axis=1).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, alpha=0.25, levels=np.arange(zz.max() + 2) - 0.5)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k", alpha=0.9)
plt.title("Frontera de decisión — Moons + MLP")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(*scatter.legend_elements(), title="Clase", loc="upper right")
plt.tight_layout()
plt.show()


# =============================================================================
# PARTE 2 — CLASIFICACIÓN (Real: Iris) con MLP
# =============================================================================
print("\n=== PARTE 2 — Clasificación (Real: Iris) ===")

iris = load_iris()
X, y = iris.data, iris.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)

scaler2 = StandardScaler()
Xtr_s = scaler2.fit_transform(Xtr)
Xte_s = scaler2.transform(Xte)

model_iris = models.Sequential()
model_iris.add(layers.Input(shape=(Xtr_s.shape[1],)))
model_iris.add(layers.Dense(64, activation="relu"))
model_iris.add(layers.Dropout(0.15))
model_iris.add(layers.Dense(64, activation="relu"))
model_iris.add(layers.Dense(len(np.unique(y)), activation="softmax"))

model_iris.compile(optimizer=optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

cb_early2 = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
cb_rlr2 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

hist2 = model_iris.fit(Xtr_s, ytr,
                       validation_split=0.2,
                       epochs=120,
                       batch_size=32,
                       callbacks=[cb_early2, cb_rlr2],
                       verbose=0)

plt.figure(figsize=(7, 4))
plt.plot(hist2.history["loss"], label="loss (train)")
plt.plot(hist2.history["val_loss"], label="loss (val)")
plt.plot(hist2.history["accuracy"], label="acc (train)")
plt.plot(hist2.history["val_accuracy"], label="acc (val)")
plt.title("Iris — MLP (curvas de aprendizaje)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

ypred = np.argmax(model_iris.predict(Xte_s, verbose=0), axis=1)
acc = accuracy_score(yte, ypred)
print(f"Accuracy (test): {acc:.3f}")

fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_predictions(yte, ypred, ax=ax, cmap="Blues")
ax.set_title("Matriz de confusión — Iris")
plt.tight_layout()
plt.show()

# =============================================================================
# PARTE 3 — REGRESIÓN (Simulado) con MLP
# =============================================================================
print("\n=== PARTE 3 — Regresión (Simulado: y = sin(x) + ruido) ===")

# 3.1 Datos simulados
rng = np.random.default_rng(0)
X = np.linspace(-3*np.pi, 3*np.pi, 600).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + rng.normal(0, 0.2, size=y_true.shape)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

scaler3 = StandardScaler()
Xtr_s = scaler3.fit_transform(Xtr)
Xte_s = scaler3.transform(Xte)

# 3.2 MLP para regresión
model_reg = models.Sequential()
model_reg.add(layers.Input(shape=(Xtr_s.shape[1],)))
model_reg.add(layers.Dense(64, activation="relu"))
model_reg.add(layers.Dropout(0.1))
model_reg.add(layers.Dense(64, activation="relu"))
model_reg.add(layers.Dense(64, activation="relu"))
model_reg.add(layers.Dense(1, activation="linear"))  # salida continua

model_reg.compile(optimizer=optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])

# cb_early3 = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
# cb_rlr3 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

hist3 = model_reg.fit(Xtr_s, ytr,
                      validation_split=0.2,
                      epochs=120,
                      batch_size=64,
                      # callbacks=[cb_early3, cb_rlr3],
                      verbose=0)

plt.figure(figsize=(7, 4))
plt.plot(hist3.history["loss"], label="loss (train)")
plt.plot(hist3.history["val_loss"], label="loss (val)")
plt.plot(hist3.history["mae"], label="mae (train)")
plt.plot(hist3.history["val_mae"], label="mae (val)")
plt.title("Regresión — MLP (curvas de aprendizaje)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

yhat = model_reg.predict(Xte_s, verbose=0).ravel()
rmse = mean_squared_error(yte, yhat)
r2 = r2_score(yte, yhat)
print(f"RMSE (test): {rmse:.3f} | R² (test): {r2:.3f}")

# Visualización de ajuste sobre todo el dominio
Xs = scaler3.transform(X)
yhat_all = model_reg.predict(Xs, verbose=0).ravel()
plt.figure(figsize=(7, 4))
plt.scatter(Xtr, ytr, s=12, alpha=0.4, label="train")
plt.scatter(Xte, yte, s=12, alpha=0.4, label="test")
plt.plot(X, y_true, lw=2, label="verdadero sin(x)")
plt.plot(X, yhat_all, lw=2, label="MLP predicción")
plt.title("Regresión 1D: sin(x) + ruido")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# PARTE 4 — REGRESIÓN (Real: California Housing) con MLP
# =============================================================================
print("\n=== PARTE 4 — Regresión (Real: California Housing) ===")

cal = fetch_california_housing()
X, y = cal.data, cal.target  # target: mediana valor de vivienda (aprox. cientos de miles)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

scaler4 = StandardScaler()
Xtr_s = scaler4.fit_transform(Xtr)
Xte_s = scaler4.transform(Xte)

model_cal = models.Sequential()
model_cal.add(layers.Input(shape=(Xtr_s.shape[1],)))
model_cal.add(layers.Dense(128, activation="relu"))
model_cal.add(layers.Dropout(0.1))
model_cal.add(layers.Dense(64, activation="relu"))
model_cal.add(layers.Dense(1, activation="linear"))

model_cal.compile(optimizer=optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=["mae"])

cb_early4 = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
cb_rlr4 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

hist4 = model_cal.fit(Xtr_s, ytr,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=128,
                      callbacks=[cb_early4, cb_rlr4],
                      verbose=0)

plt.figure(figsize=(7, 4))
plt.plot(hist4.history["loss"], label="loss (train)")
plt.plot(hist4.history["val_loss"], label="loss (val)")
plt.plot(hist4.history["mae"], label="mae (train)")
plt.plot(hist4.history["val_mae"], label="mae (val)")
plt.title("California Housing — MLP (curvas de aprendizaje)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

yhat = model_cal.predict(Xte_s, verbose=0).ravel()
rmse = mean_squared_error(yte, yhat)
r2 = r2_score(yte, yhat)
print(f"RMSE (test): {rmse:.3f} | R² (test): {r2:.3f}")

# =============================================================================
# PARTE 5 — CLASIFICACIÓN DE IMÁGENES (MNIST) con CNN
# =============================================================================
print("\n=== PARTE 5 — Clasificación de Imágenes (MNIST) con CNN ===")

# 5.1 Cargamos MNIST (se descarga automáticamente la primera vez)
(Xtr, ytr), (Xte, yte) = tf.keras.datasets.mnist.load_data()

# 5.2 Normalizamos a [0,1] y añadimos canal (28,28,1)
Xtr = (Xtr.astype("float32") / 255.0)[..., np.newaxis]
Xte = (Xte.astype("float32") / 255.0)[..., np.newaxis]

# 5.3 Definimos una CNN simple
cnn = models.Sequential()
cnn.add(layers.Input(shape=Xtr.shape[1:]))
cnn.add(layers.Conv2D(32, kernel_size=3, activation="relu"))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Conv2D(64, kernel_size=3, activation="relu"))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Flatten())
cnn.add(layers.Dense(128, activation="relu"))
cnn.add(layers.Dropout(0.25))
cnn.add(layers.Dense(10, activation="softmax"))

cnn.compile(optimizer=optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

cb_early5 = callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
cb_rlr5 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

hist5 = cnn.fit(Xtr, ytr,
                validation_split=0.15,
                epochs=6,        # pequeño para CPU
                batch_size=128,
                callbacks=[cb_early5, cb_rlr5],
                verbose=0)

plt.figure(figsize=(7, 4))
plt.plot(hist5.history["loss"], label="loss (train)")
plt.plot(hist5.history["val_loss"], label="loss (val)")
plt.plot(hist5.history["accuracy"], label="acc (train)")
plt.plot(hist5.history["val_accuracy"], label="acc (val)")
plt.title("MNIST — CNN (curvas de aprendizaje)")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

test_loss, test_acc = cnn.evaluate(Xte, yte, verbose=0)
print(f"Accuracy (test): {test_acc:.3f}")

# 5.4 Predicciones de ejemplo
idx = np.random.choice(len(Xte), size=16, replace=False)
imgs = Xte[idx]
preds = np.argmax(cnn.predict(imgs, verbose=0), axis=1)
true = yte[idx]

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for ax, im, p, t in zip(axes.ravel(), imgs, preds, true):
    ax.imshow(im.squeeze(), cmap="gray")
    ax.axis("off")
    ax.set_title(f"pred:{p} | true:{t}", fontsize=8)
fig.suptitle("Predicciones — MNIST")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# IDEAS DE EJERCICIOS:
#  - Cambia el número de neuronas/capas; observa overfitting/underfitting.
#  - Prueba otras activaciones: 'tanh', 'gelu'.
#  - Añade BatchNormalization después de capas densas/conv.
#  - Cambia el optimizador (SGD, RMSprop) y su learning rate.
#  - En MNIST, agrega capas de data augmentation: layers.RandomRotation(0.1), etc.
#  - Guarda y recarga modelos: model.save("modelo.keras"); tf.keras.models.load_model(...).
#  - Ajusta "monitor" de EarlyStopping para priorizar exactitud vs pérdida.
# -----------------------------------------------------------------------------
