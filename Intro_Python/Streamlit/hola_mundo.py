import streamlit as st

st.title("Hola, Streamlit 🚀")
st.write("Esta es mi primera app web con Python.")

x = st.slider("Elige un número", 0, 100, 42)
st.write(f"Tu número al cuadrado es: {x**2}")

