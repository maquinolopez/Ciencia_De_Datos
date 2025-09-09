#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================
# Práctica en clase: Neotoma "organic carbon"
# =========================================================
# Lee el CSV descargado desde Neotoma (dataset type "organic carbon")
#
# Requisitos:
#   pip install pandas numpy matplotlib scikit-learn
# ---------------------------------------------------------

# Cambiemos el directorio de trabajo
from os import chdir
chdir("directoio_donde esta el codigo")
   
# Cargar pandas
import pandas as pd

pd.set_option("display.max_columns", None)  # Mostrar todas las columnas
pd.set_option("display.max_rows", None)     # Mostrar todas las filas
pd.set_option("display.width", None)        # Ajuste automático al ancho de la terminal
pd.set_option("display.max_colwidth", None) # No truncar el contenido de celdas


# Ajusta esta ruta a tu archivo CSV descargado:
file_path = "./Data/dataset15957_site10443.csv"  # <- cámbiala si tu archivo tiene otro nombre
# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# Leemos el archivo CSV con pandas. 
# Esto nos da un DataFrame con las características en filas y las muestras en columnas.
df = pd.read_csv(file_path)
print(df)
print(df.head())

# Transponemos el DataFrame: las filas pasan a ser columnas y las columnas filas.
# Usamos la columna 'name' como índice, de forma que sus valores (Depth, Thickness, etc.)
# se conviertan en nombres de columna después de la transposición.
df_t = df.set_index("name").T

# Reiniciamos el índice para que las etiquetas de muestra (S153951, S153952, …)
# no queden como índice, sino como una columna llamada "Sample".
df_t = df_t.reset_index().rename(columns={"index": "Sample"})

# Imprimimos el DataFrame transpuesto para ver su estructura inicial.
print(df_t.head())

# Notemos que las primeras 4 columnas corresponden a características que no nos sirven
# (group, element, units, context). Aquí las eliminamos.
df_t = df_t.drop(df.index[0:4])

# Imprimimos los nombres de las columnas que nos quedan para confirmar la limpieza.
print(df_t.columns)

# Finalmente, removemos otras columnas que contienen metadatos o información redundante
# (AnalysisUnitName, Thickness, Sample Name, Sample ID, Neotoma 1, Loisel et al. 2014)
df_t = df_t.drop(columns=['AnalysisUnitName', 'Thickness', 'Sample Name',
       'Sample ID', 'Neotoma 1', 'Loisel et al. 2014'])

# Imprimimos el DataFrame final, ya limpio y listo para análisis.
print(df_t.head())

# ==========================================================
# Exploración inicial de la base de datos df_t
# ==========================================================

# 1) Revisemos los tipos de datos de cada columna
print("\n--- Tipos de datos ---")
print(df_t.dtypes)

# Convertimos todas las columnas excepto la primera a numéricas
df_t.iloc[:, 1:] = df_t.iloc[:, 1:].apply(pd.to_numeric)

# Verificamos que efectivamente ya son numéricas
print(df_t.dtypes)

# 2) Verifiquemos si existen valores faltantes en las columnas
print("\n--- Valores faltantes por columna ---")
print(df_t.isnull().sum())

# 3) Revisemos si hay filas duplicadas
print("\n--- Número de filas duplicadas ---")
print(df_t.duplicated().sum())

# 4) Obtenemos un resumen estadístico de las columnas numéricas
print("\n--- Resumen estadístico de columnas numéricas ---")
print(df_t.describe())

# 5) También revisamos un resumen de columnas categóricas (tipo objeto)
print("\n--- Resumen de columnas categóricas ---")
print(df_t.describe(include=['object']))

# 6) Mostramos la dimensión de la base de datos
print("\n--- Dimensión de la base de datos (filas, columnas) ---")
print(df_t.shape)


# ==========================================================
# Exploracion grafica inicial
# ==========================================================
import matplotlib.pyplot as plt
import seaborn as sns

# --- Selección de 4 variables de interés ---
vars_interest = ["bulk density", "loss-on-ignition", "organic carbon density", "organic matter density"]

# Aseguramos que sean numéricas
for col in vars_interest:
    df_t[col] = pd.to_numeric(df_t[col], errors="coerce")

# --- Gráfica compuesta ---
fig, axes = plt.subplots(3, len(vars_interest), figsize=(16, 10))

for i, col in enumerate(vars_interest):
    # Histograma
    sns.histplot(df_t[col].dropna(), bins=30, kde=False, ax=axes[0, i], color="skyblue")
    axes[0, i].set_title(f"Histograma: {col}")
    
    # Densidad kernel
    sns.kdeplot(df_t[col].dropna(), ax=axes[1, i], color="green", fill=True)
    axes[1, i].set_title(f"Densidad kernel: {col}")
    
    # Boxplot
    sns.boxplot(y=df_t[col].dropna(), ax=axes[2, i], color="orange")
    axes[2, i].set_title(f"Boxplot: {col}")

plt.tight_layout()
plt.show()



# ==========================================================
# Exploración inicial si hubiera datos faltantes
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1) Insertar datos faltantes al 3% en columnas numéricas
# ---------------------------------------------------------
rng = np.random.default_rng(seed=123)
arr = df_t[df_t.columns[2:]].to_numpy(copy=True).astype(float)
mask_missing = rng.random(arr.shape) < 0.03
arr[mask_missing] = np.nan
df_with_na = df_t.copy()
df_with_na[df_t.columns[2:]] = arr

# Verifiquemos si existen valores faltantes en las columnas
print("\n--- Valores faltantes por columna ---")
print(df_with_na.isnull().sum())


# ---------------------------------------------------------
# 2a) Imputación simple por la media de cada columna
# ---------------------------------------------------------
df_mean_imp = df_with_na.copy()
for col in df_t.columns[2:]:
    mean_val = df_mean_imp[col].mean(skipna=True)
    df_mean_imp[col] = df_mean_imp[col].fillna(mean_val)

# ---------------------------------------------------------
# 2b) Imputación por muestreo de una Normal(mean, sd) de la columna
# ---------------------------------------------------------
df_norm_imp = df_with_na.copy()
for col in df_t.columns[2:]:
    col_mean = df_norm_imp[col].mean(skipna=True)
    col_std = df_norm_imp[col].std(skipna=True)
    # indices donde hay NA
    na_idx = df_norm_imp[col].isna()
    n_missing = na_idx.sum()
    if n_missing > 0:
        simulated_vals = rng.normal(col_mean, col_std, n_missing)
        df_norm_imp.loc[na_idx, col] = simulated_vals

# ---------------------------------------------------------
# 3) Histogramas comparativos
# ---------------------------------------------------------
# Seleccionamos columnas numéricas sin 'Depth'
cols_to_plot = df_t.columns[2:]  # quitar Sample y Depth
ncols = 4
nrows = 2
nplots = min(len(cols_to_plot), ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot[:nplots]):
    # Arriba: Original vs Media
    ax = axes[i]
    ax.hist(df_t[col].dropna(), bins=30, alpha=0.5, label="Original")
    ax.hist(df_mean_imp[col], bins=30, alpha=0.5, label="Imputación media")
    ax.set_title(f"{col} (Original vs Media)")
    ax.legend(fontsize=8)
    
    # Abajo: Original vs Normal(mu, sigma)
    ax2 = axes[i + nplots]
    ax2.hist(df_t[col].dropna(), bins=30, alpha=0.5, label="Original")
    ax2.hist(df_norm_imp[col], bins=30, alpha=0.5, label="Imputación N(μ,σ)")
    ax2.set_title(f"{col} (Original vs Normal)")
    ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------

# --- Selección de variables ---
y_col = 'organic carbon density'   # <- aquí cambias la variable de interés
x_col = "Depth"

# Extraer datos (quitando NA)
df_sub = df_t[[x_col, y_col]].dropna()
X = df_sub[[x_col]].to_numpy(dtype=float)
y = df_sub[y_col].to_numpy(dtype=float)

# Añadir intercepto
X1 = np.column_stack([np.ones(X.shape[0]), X])

# Calcular beta con mínimos cuadrados
beta_hat, *_ = np.linalg.lstsq(X1, y, rcond=None)

# Calcular matriz sombrero H
H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
leverages = np.diag(H)

# Regla práctica de corte
n, p = X1.shape
threshold = 2*p/n

# --- Visualización ---
plt.figure(figsize=(8,5))
plt.scatter(X, y, c="blue", alpha=0.6, label="Datos")
plt.plot(X, X1 @ beta_hat, c="red", label="Recta ajustada")

# Resaltar puntos con leverage alto
outliers = leverages > threshold
# plt.scatter(X[outliers], y[outliers], facecolors="none", edgecolors="r", s=100, label="Posible outlier")

# plt.xlabel(x_col)
# plt.ylabel(y_col)
# plt.title(f"Identificación de outliers con Hat Matrix ({y_col})")
# plt.legend()
# plt.show()



# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# Variables a graficar (todas excepto Sample y Depth)
vars_to_plot = [c for c in df_t.columns if c not in ["Sample", "Depth"]]

ncols = 2
nrows = (len(vars_to_plot) + 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
axes = axes.flatten()

for i, col in enumerate(vars_to_plot):
    ax = axes[i]
    ax.plot(df_t["Depth"], pd.to_numeric(df_t[col]) ,
               alpha=0.6, linewidth=1.1)
    ax.set_xlabel("Depth")
    ax.set_ylabel(col)
    ax.set_title(f"Depth vs {col}")

# Eliminar ejes sobrantes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# --- Selección de las 4 variables ---
vars_interest = ["bulk density", "loss-on-ignition", "organic carbon density", "organic matter density"]


# --- Pairplot ---
sns.pairplot(df_t[vars_interest], diag_kind="kde", corner=True,
             plot_kws={"alpha":0.6, "edgecolor":"k"})
plt.suptitle("Pairplot de variables de interés", y=1.02, fontsize=14)
plt.show()

# --- Heatmap de correlaciones ---
corr_matrix = df_t[vars_interest].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f",
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Mapa de calor de correlaciones", fontsize=14)
plt.show()
















