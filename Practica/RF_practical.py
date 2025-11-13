#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:21:54 2025

@author: maquinol
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import seaborn as sns

# Configuración
plt.rcParams['font.size'] = 12
np.random.seed(42)

print("=" * 80)
print("TUTORIAL: ÁRBOLES DE DECISIÓN, BAGGING Y RANDOM FOREST CON SCIKIT-LEARN")
print("=" * 80)

# =============================================================================
# PARTE 1: CLASIFICACIÓN
# =============================================================================
print("\n" + "=" * 60)
print("PARTE 1: PROBLEMA DE CLASIFICACIÓN")
print("=" * 60)

# 1) Datos
print("\n1. Generando datos de clasificación...")
X_clf, y_clf = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

print(f"   Forma de los datos: {X_clf.shape}")
print(f"   Clases: {np.unique(y_clf)}")
print(f"   Distribución: {np.bincount(y_clf)}")

# 2) Visual: datos
def plot_decision_boundary(clf, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_title(title)

plt.figure(figsize=(20, 5))
ax1 = plt.subplot(1, 4, 1)
ax1.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='viridis', alpha=0.7, edgecolors='k')
ax1.set_xlabel('Característica 1')
ax1.set_ylabel('Característica 2')
ax1.set_title('Datos de Clasificación\n(2 clases, 2 características)')
plt.colorbar(ax1.collections[0], ax=ax1, label='Clase')

# =============================================================================
# MODELOS DE CLASIFICACIÓN
# =============================================================================
print("\n2. Entrenando modelos de clasificación...")

# Árbol de Decisión (base learner)
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_clf_train, y_clf_train)
y_dt_pred = dt_clf.predict(X_clf_test)
accuracy_dt = accuracy_score(y_clf_test, y_dt_pred)

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),  # <── cambio aquí
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42
)
bag_clf.fit(X_clf_train, y_clf_train)
y_bag_pred = bag_clf.predict(X_clf_test)
accuracy_bag = accuracy_score(y_clf_test, y_bag_pred)

# Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_clf.fit(X_clf_train, y_clf_train)
y_rf_pred = rf_clf.predict(X_clf_test)
accuracy_rf = accuracy_score(y_clf_test, y_rf_pred)

print(f"   ✓ Árbol de Decisión:  acc = {accuracy_dt:.4f}")
print(f"   ✓ Bagging (árboles):  acc = {accuracy_bag:.4f} | OOB = {bag_clf.oob_score_:.4f}")
print(f"   ✓ Random Forest:      acc = {accuracy_rf:.4f}")

# 3) Visual: fronteras de decisión (Árbol, Bagging, RF)
ax2 = plt.subplot(1, 4, 2)
plot_decision_boundary(dt_clf, X_clf_test, y_clf_test, ax2,
                       f'Árbol de Decisión\nAccuracy: {accuracy_dt:.3f}')

ax3 = plt.subplot(1, 4, 3)
plot_decision_boundary(bag_clf, X_clf_test, y_clf_test, ax3,
                       f'Bagging (50 árboles)\nAcc: {accuracy_bag:.3f} | OOB: {bag_clf.oob_score_:.3f}')

ax4 = plt.subplot(1, 4, 4)
plot_decision_boundary(rf_clf, X_clf_test, y_clf_test, ax4,
                       f'Random Forest (50)\nAccuracy: {accuracy_rf:.3f}')

plt.tight_layout()
plt.savefig('clasificacion_comparacion_con_bagging.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# COMPARACIÓN DETALLADA CLASIFICACIÓN
# =============================================================================
print("\n3. Comparación detallada - Clasificación:")

# Validación cruzada
models_clf = {
    'Árbol Decisión': dt_clf,
    'Bagging': bag_clf,
    'Random Forest': rf_clf
}

cv_scores = {}
for name, model in models_clf.items():
    scores = cross_val_score(model, X_clf, y_clf, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    print(f"   {name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Gráfica de comparación
plt.figure(figsize=(10, 6))
boxes = plt.boxplot(cv_scores.values(), labels=cv_scores.keys(), patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(boxes['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Accuracy')
plt.title('Comparación de Modelos - Validación Cruzada (5 folds)')
plt.grid(True, alpha=0.3)
plt.savefig('clasificacion_cv_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# PARTE 2: REGRESIÓN
# =============================================================================
print("\n" + "=" * 60)
print("PARTE 2: PROBLEMA DE REGRESIÓN")
print("=" * 60)

# Generar datos de regresión
print("\n1. Generando datos de regresión...")
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=1,
    noise=20,
    random_state=42
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

print(f"   Forma de los datos: {X_reg.shape}")
print(f"   Rango de valores Y: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

# =============================================================================
# MODELOS DE REGRESIÓN
# =============================================================================
print("\n2. Entrenando modelos de regresión...")

# Árbol de Regresión
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_reg_train, y_reg_train)
y_dt_reg_pred = dt_reg.predict(X_reg_test)
mse_dt = mean_squared_error(y_reg_test, y_dt_reg_pred)
r2_dt = r2_score(y_reg_test, y_dt_reg_pred)

# Bagging Regressor
bag_reg = BaggingRegressor(
    DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    max_samples=0.8,
    random_state=42
)
bag_reg.fit(X_reg_train, y_reg_train)
y_bag_reg_pred = bag_reg.predict(X_reg_test)
mse_bag = mean_squared_error(y_reg_test, y_bag_reg_pred)
r2_bag = r2_score(y_reg_test, y_bag_reg_pred)

# Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_reg.fit(X_reg_train, y_reg_train)
y_rf_reg_pred = rf_reg.predict(X_reg_test)
mse_rf = mean_squared_error(y_reg_test, y_rf_reg_pred)
r2_rf = r2_score(y_reg_test, y_rf_reg_pred)

print(f"   ✓ Árbol de Regresión - MSE: {mse_dt:.2f}, R²: {r2_dt:.4f}")
print(f"   ✓ Bagging Regressor - MSE: {mse_bag:.2f}, R²: {r2_bag:.4f}")
print(f"   ✓ Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.4f}")

# Visualizar resultados de regresión
plt.figure(figsize=(15, 5))

# Ordenar datos para plotting
sort_idx = np.argsort(X_reg_test.flatten())
X_sorted = X_reg_test[sort_idx]
y_true_sorted = y_reg_test[sort_idx]

plt.subplot(1, 3, 1)
plt.scatter(X_reg, y_reg, alpha=0.3, label='Datos')
plt.plot(X_sorted, y_dt_reg_pred[sort_idx], 'r-', linewidth=2, label='Predicción')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Árbol de Regresión\nMSE: {mse_dt:.2f}, R²: {r2_dt:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(X_reg, y_reg, alpha=0.3, label='Datos')
plt.plot(X_sorted, y_bag_reg_pred[sort_idx], 'g-', linewidth=2, label='Predicción')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Bagging Regressor\nMSE: {mse_bag:.2f}, R²: {r2_bag:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(X_reg, y_reg, alpha=0.3, label='Datos')
plt.plot(X_sorted, y_rf_reg_pred[sort_idx], 'b-', linewidth=2, label='Predicción')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Random Forest\nMSE: {mse_rf:.2f}, R²: {r2_rf:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regresion_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# COMPARACIÓN DETALLADA REGRESIÓN
# =============================================================================
print("\n3. Comparación detallada - Regresión:")

models_reg = {
    'Árbol Regresión': dt_reg,
    'Bagging': bag_reg,
    'Random Forest': rf_reg
}

cv_scores_reg = {}
for name, model in models_reg.items():
    scores = cross_val_score(model, X_reg, y_reg, cv=10, scoring='r2')
    cv_scores_reg[name] = scores
    print(f"   {name} - R²: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Gráfica de comparación R²
plt.figure(figsize=(10, 6))
boxes = plt.boxplot(cv_scores_reg.values(), labels=cv_scores_reg.keys(), patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(boxes['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('R² Score')
plt.title('Comparación de Modelos de Regresión - Validación Cruzada (5 folds)')
plt.grid(True, alpha=0.3)
plt.savefig('regresion_cv_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# PARTE 3: EJEMPLO PRÁCTICO - DATOS COMPLEJOS
# =============================================================================
print("\n" + "=" * 60)
print("PARTE 3: EJEMPLO PRÁCTICO - DATOS COMPLEJOS (Moons)")
print("=" * 60)

# Datos complejos no lineales
X_moons, y_moons = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

print(f"Datos Moons - Forma: {X_moons.shape}")

# Comparación en datos complejos
models_moons = {
    'Árbol Simple': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Árbol Profundo': DecisionTreeClassifier(max_depth=20, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=50, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
}

results_moons = {}
plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(models_moons.items(), 1):
    model.fit(X_m_train, y_m_train)
    y_pred = model.predict(X_m_test)
    accuracy = accuracy_score(y_m_test, y_pred)
    results_moons[name] = accuracy
    
    plt.subplot(2, 2, i)
    plot_decision_boundary(model, X_m_test, y_m_test, plt.gca(), 
                          f'{name}\nAccuracy: {accuracy:.3f}')

plt.tight_layout()
plt.savefig('moons_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nResultados en datos Moons (no lineales):")
for name, acc in results_moons.items():
    print(f"   {name}: {acc:.4f}")

# =============================================================================
# PARTE 4: RESUMEN Y RECOMENDACIONES
# =============================================================================
print("\n" + "=" * 60)
print("PARTE 4: RESUMEN Y RECOMENDACIONES")
print("=" * 60)

print("""
RESUMEN DE MODELOS:

1. ÁRBOL DE DECISIÓN:
   - ✅ Ventajas: Fácil de interpretar, rápido de entrenar
   - ❌ Desventajas: Propenso a sobreajuste, alta varianza
   - 📊 Uso: Datos pequeños, interpretabilidad importante

2. BAGGING (Bootstrap Aggregating):
   - ✅ Ventajas: Reduce varianza, más estable que árbol simple
   - ❌ Desventajas: Menos interpretable, requiere más cómputo
   - 📊 Uso: Cuando se quiere mejorar un modelo base estable

3. RANDOM FOREST:
   - ✅ Ventajas: Reduce varianza y sobreajuste, robusto
   - ❌ Desventajas: Menos interpretable, hiperparámetros a ajustar
   - 📊 Uso: Problemas generales, buen rendimiento out-of-the-box

PARÁMETROS IMPORTANTES:

• Árbol de Decisión:
  - max_depth: Profundidad máxima (controla complejidad)
  - min_samples_split: Mínimo muestras para dividir nodo
  - min_samples_leaf: Mínimo muestras en hoja

• Bagging:
  - n_estimators: Número de modelos base
  - max_samples: Fracción de muestras por modelo
  - base_estimator: Modelo base a usar

• Random Forest:
  - n_estimators: Número de árboles
  - max_depth: Profundidad de árboles
  - max_features: Características por división
  - min_samples_split: Mínimo para dividir

BUENAS PRÁCTICAS:

1. Siempre usar validación cruzada
2. Comenzar con Random Forest como baseline
3. Ajustar hiperparámetros con GridSearchCV
4. Considerar interpretabilidad vs rendimiento
5. Usar Árbol simple para entender los datos
""")

# =============================================================================
# EJEMPLO DE OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================
print("\n" + "=" * 60)
print("EJEMPLO: OPTIMIZACIÓN DE RANDOM FOREST")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

# Búsqueda de grilla simple
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 7, 15, None],
    'min_samples_split': [2, 5, 10,100]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_clf_train, y_clf_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score: {grid_search.best_score_:.4f}")

# Comparación antes/después de optimización
rf_base = RandomForestClassifier(n_estimators=50, random_state=42)
rf_base.fit(X_clf_train, y_clf_train)
y_base_pred = rf_base.predict(X_clf_test)
accuracy_base = accuracy_score(y_clf_test, y_base_pred)

rf_optimized = grid_search.best_estimator_
y_opt_pred = rf_optimized.predict(X_clf_test)
accuracy_opt = accuracy_score(y_clf_test, y_opt_pred)

print(f"\nComparación Random Forest:")
print(f"   Base (n_estimators=50): {accuracy_base:.4f}")
print(f"   Optimizado: {accuracy_opt:.4f}")
print(f"   Mejora: {(accuracy_opt - accuracy_base)*100:.2f}%")


# Mostrar feature importance
plt.figure(figsize=(10, 6))
feature_importance = rf_clf.feature_importances_
features = ['Feature 1', 'Feature 2']
plt.bar(features, feature_importance)
plt.title('Feature Importance - Random Forest')
plt.ylabel('Importancia')
plt.grid(True, alpha=0.3)
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Feature Importance del Random Forest:")
for feat, imp in zip(features, feature_importance):
    print(f"   {feat}: {imp:.4f}")