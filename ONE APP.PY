import streamlit as st
import numpy as np
import pandas as pd
import math
import io
import re
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.set_page_config(page_title="Combined Streamlit App", layout="wide")

# Title of the app
st.title("Combined Streamlit Application")

# Create tabs for the two applications
tab1, tab2 = st.tabs(["Azimuth to Coordinates Converter", "Point Alignment Check"])

# --- Azimuth to Coordinates Converter
with tab1:
    # Add hit counter image
    st.markdown(
        '<img src="https://hitscounter.dev/api/hit?url=https%3a%2f%2fpolar2xy.streamlit.app%2f&label=visitas&icon=github&color=%233dd5f3&message=&style=flat&tz=utc">',
        unsafe_allow_html=True
    )

    # Language translations
    translations = {...}  # Keep the translations dictionary from the original code

    def get_text(key, lang='en'):
        """Get translated text for the given key and language"""
        return translations.get(lang, translations['en']).get(key, key)

    # Existing function definitions and main function for the first app...

    def main_azimuth_converter():
        st.sidebar.header(get_text('settings', 'en'))  # Replace 'en' with selected language

        # Initialize language in session state
        if 'language' not in st.session_state:
            st.session_state.language = 'en'

        # Language selector in sidebar
        lang = st.session_state.language
        # Add your existing logic for the azimuth converter here...

    main_azimuth_converter()

# --- Point Alignment Check
with tab2:
    st.title("📐 Verificación de Alineación de un Punto con la Línea AB")
    st.markdown("Introduce las coordenadas de dos puntos **A y B** y un punto **PT** (punto de trabajo).")

    # --- Input from user
    xa = float(st.text_input("Cord X A", value="1072.998"))
    ya = float(st.text_input("Cord Y A", value="971.948"))
    xb = float(st.text_input("Cord X B", value="963.595"))
    yb = float(st.text_input("Cord Y B", value="1012.893"))
    xpt = float(st.text_input("Cord X PI", value="1040.749"))
    ypt = float(st.text_input("Cord Y PI", value="983.875"))
    tol = float(st.text_input("Tolerancia (m)", value="0.01"))

    # --- Functions
    def distancia_perpendicular(a, b, pt):
        (xa, ya), (xb, yb), (xpt, ypt) = a, b, pt
        det = (xb - xa)*(ya - ypt) - (yb - ya)*(xa - xpt)
        ab = np.sqrt((xb - xa)**2 + (yb - ya)**2)
        d = -det / ab  # Positive = right, negative = left
        return d

    def proyeccion(a, b, pt):
        a = np.array(a)
        b = np.array(b)
        pt = np.array(pt)
        ab = b - a
        ap = pt - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        return a + t * ab

    # --- Calculations
    a = (xa, ya)
    b = (xb, yb)
    pt = (xpt, ypt)

    d_signed = distancia_perpendicular(a, b, pt)
    d_abs = abs(d_signed)
    proj = proyeccion(a, b, pt)
    corr_vector = proj - np.array(pt)
    alineado = d_abs <= tol
    dist_perp = np.sqrt((proj[0] - xpt)**2 + (proj[1] - ypt)**2)

    # --- Results
    st.subheader("📊 Resultados")
    st.write(f"Distancia perpendicular (con signo) = **{d_signed:.3f} m**")
    if d_signed > 0:
        st.success("➡️ PT está a la **derecha** de la línea AB")
    elif d_signed < 0:
        st.warning("⬅️ PT está a la **izquierda** de la línea AB")
    else:
        st.info("🎯 PT está exactamente sobre la línea AB")

    st.write(f"Coordenadas de la proyección sobre AB: **({proj[0]:.3f}, {proj[1]:.3f})**")
    st.write(f"Vector de corrección: δx = {corr_vector[0]:.3f}, δy = {corr_vector[1]:.3f}")

    # --- Visualization
    st.subheader("📈 Visualización Mejorada")
    fig, ax = plt.subplots(figsize=(8, 8))  # Size of the chart

    # Plot line AB
    ax.plot([xa, xb], [ya, yb], 'b-', linewidth=0.5, label="Línea AB")
    # Plot perpendicular line
    ax.plot([xpt, proj[0]], [ypt, proj[1]], 'r--', linewidth=2, label="Perpendicular")

    # We can plot the points similarly as in your provided code
    ax.plot(xpt, ypt, 'ro', markersize=12, markerfacecolor='none', label="PT")
    ax.plot(proj[0], proj[1], 'go', markersize=12, markerfacecolor='none', label="Proyección de PT")

    # More plotting logic...

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Alineación de PT respecto a AB", fontsize=12)
    ax.grid(True)
    ax.axis("equal")
    ax.legend(fontsize=9)

    st.pyplot(fig)
    if alineado:

        st.success(f"✅ PT está alineado con AB dentro de la tolerancia de {tol} m")
