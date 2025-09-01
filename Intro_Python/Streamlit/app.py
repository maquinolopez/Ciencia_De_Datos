# app_basico.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("🌟 Mi primera app en Streamlit")

st.write("Esta aplicación muestra lo básico de Streamlit: texto, controles, tablas y un gráfico.")

# ==============================
# Barra lateral
# ==============================
st.sidebar.header("Controles")
numero = st.sidebar.slider("Elige un número", 0, 100, 50)

st.write(f"El número seleccionado es **{numero}**, su cuadrado es **{numero**2}**.")

# ==============================
# Datos de ejemplo o CSV
# ==============================
st.subheader("📂 Carga de datos")
archivo = st.file_uploader("Sube un CSV", type="csv")

if archivo:
    df = pd.read_csv(archivo)
else:
    st.info("No subiste archivo, usamos datos de ejemplo.")
    df = pd.DataFrame({
        "x": np.arange(1, 11),
        "y": np.random.randint(1, 20, 10)
    })

st.dataframe(df)

# ==============================
# Gráfico sencillo con Altair
# ==============================
st.subheader("📈 Gráfico")
grafico = alt.Chart(df).mark_line(point=True).encode(
    x="x",
    y="y"
)
st.altair_chart(grafico, use_container_width=True)
