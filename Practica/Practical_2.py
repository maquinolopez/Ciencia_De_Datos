#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 14:45:24 2025

@author: MAL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar dataset Iris (usamos solo 2 variables para poder graficar)
iris = load_iris()
X = iris.data[:, :2]   # longitud y ancho de sépalo
y = iris.target
target_names = iris.target_names

# DataFrame para exploración
df = pd.DataFrame(X, columns=iris.feature_names[:2])
df["species"] = [target_names[i] for i in y]
print("Primeras filas del dataset Iris:")
print(df.head())

# Visualizar en 2D
plt.figure(figsize=(6,5))
for i, t in enumerate(target_names):
    plt.scatter(
        df.loc[df["species"]==t, iris.feature_names[0]], 
        df.loc[df["species"]==t, iris.feature_names[1]], 
        label=t
    )
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title("Iris dataset (2 variables)")
plt.show()

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Tamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)

def plot_decision_boundary(model, X, y, title):
    # Crear nueva figura cada vez
    plt.figure(figsize=(6, 5))
    
    # Crear grid en el espacio 2D
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predicciones sobre el grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar fronteras y puntos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k", s=40)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)
    plt.show()


#=====================================
#======  Naive Bayes =================
#=====================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Entrenar
nb = GaussianNB()
nb.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(nb, X_train, y_train, "Frontera de decisión - Naive Bayes")

# Evaluar
y_pred_nb = nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_nb)
print("Matriz de confusión (Naive Bayes):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precisión:", precision_score(y_test, y_pred_nb, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_nb, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_nb, average="weighted"))





#=====================================
#=LDA (Linear Discriminant Analysis)=
#=====================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Entrenar
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(lda, X_train, y_train, "Frontera de decisión - LDA")

# Evaluar
y_pred_lda = lda.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lda)
print("Matriz de confusión (LDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_lda))
print("Precisión:", precision_score(y_test, y_pred_lda, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_lda, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_lda, average='weighted'))

#=====================================
# QDA (Quadratic Discriminant Analysis) =
#=====================================
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Entrenar
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(qda, X_train, y_train, "Frontera de decisión - QDA")

# Evaluar
y_pred_qda = qda.predict(X_test)
cm = confusion_matrix(y_test, y_pred_qda)
print("Matriz de confusión (QDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_qda))
print("Precisión:", precision_score(y_test, y_pred_qda, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_qda, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_qda, average='weighted'))




#=====================================
# k-NN (k-Nearest Neighbors) =
#=====================================
from sklearn.neighbors import KNeighborsClassifier

# Entrenar (con k=5 vecinos)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(knn, X_train, y_train, "Frontera de decisión - k-NN (k=5)")

# Evaluar
y_pred_knn = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)
print("Matriz de confusión (k-NN):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precisión:", precision_score(y_test, y_pred_knn, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_knn, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_knn, average='weighted'))


#=====================================
# Comparación de todos los modelos
#=====================================
import pandas as pd

# Guardar resultados en un diccionario
results = {
    "Naive Bayes": {
        "Acc": accuracy_score(y_test, y_pred_nb),
        "Precisión": precision_score(y_test, y_pred_nb, average="weighted"),
        "Recall": recall_score(y_test, y_pred_nb, average="weighted"),
        "F1": f1_score(y_test, y_pred_nb, average="weighted")
    },
    "LDA": {
        "Acc": accuracy_score(y_test, y_pred_lda),
        "Precisión": precision_score(y_test, y_pred_lda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_lda, average="weighted"),
        "F1": f1_score(y_test, y_pred_lda, average="weighted")
    },
    "QDA": {
        "Acc": accuracy_score(y_test, y_pred_qda),
        "Precisión": precision_score(y_test, y_pred_qda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_qda, average="weighted"),
        "F1": f1_score(y_test, y_pred_qda, average="weighted")
    },
    "k-NN (k=5)": {
        "Acc": accuracy_score(y_test, y_pred_knn),
        "Precisión": precision_score(y_test, y_pred_knn, average="weighted"),
        "Recall": recall_score(y_test, y_pred_knn, average="weighted"),
        "F1": f1_score(y_test, y_pred_knn, average="weighted")
    }
}

# Convertir a DataFrame para visualización
df_results = pd.DataFrame(results).T
print("\n=== Comparación Final de Modelos ===")
print(df_results.round(3))

#=====================================
# Validación cruzada comparativa
#=====================================

from sklearn.model_selection import StratifiedKFold, cross_val_score

models = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Validación Cruzada (5-fold, accuracy) ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:15s}: Acc = {scores.mean():.3f} ± {scores.std():.3f}")



#=====================================
# Matrices de confusión
#=====================================

import seaborn as sns

# Lista de modelos ya entrenados
trained_models = {
    "Naive Bayes": nb,
    "LDA": lda,
    "QDA": qda,
    "k-NN": knn
}

# Graficar matrices de confusión
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()








