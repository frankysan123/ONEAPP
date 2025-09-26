import streamlit as st
import numpy as np
import pandas as pd
import math
import io
import re
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Herramientas Topográficas", 
    layout="wide",
    page_icon="📐"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .division-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    /* Hide number input arrows */
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield;
    }
</style>
""", unsafe_allow_html=True)

# Language translations
TRANSLATIONS = {
    'es': {
        'tab1': '📐 Alineación PC-AB',
        'tab2': '🧭 Azimut a Coordenadas',
        'title_tab1': 'Verificación de Alineación de Punto de Control con Línea AB',
        'title_tab2': 'Convertidor de Azimut a Coordenadas',
        'subtitle_tab1': 'Introduce las coordenadas de dos puntos A y B y un punto PC (Punto de Control).',
        'subtitle_tab2': 'Convierte medidas de azimut y distancia a coordenadas X,Y.',
        'reference_point': 'Punto de Referencia',
        'reference_x': 'Referencia X',
        'reference_y': 'Referencia Y',
        'distance': 'Distancia',
        'results': 'Resultados',
        'x_coordinate': 'Coordenada X',
        'y_coordinate': 'Coordenada Y',
    },
    'en': {
        'tab1': '📐 PC-AB Alignment',
        'tab2': '🧭 Azimuth to Coordinates', 
        'title_tab1': 'Control Point Alignment Verification with Line AB',
        'title_tab2': 'Azimuth to Coordinates Converter',
        'subtitle_tab1': 'Enter coordinates for two points A and B and a control point PC.',
        'subtitle_tab2': 'Convert azimuth and distance measurements to X,Y coordinates.',
        'reference_point': 'Reference Point',
        'reference_x': 'Reference X',
        'reference_y': 'Reference Y',
        'distance': 'Distance',
        'results': 'Results',
        'x_coordinate': 'X Coordinate',
        'y_coordinate': 'Y Coordinate',
    }
}

def get_text(key, lang='es'):
    return TRANSLATIONS.get(lang, TRANSLATIONS['es']).get(key, key)

# ===== TAB 1 FUNCTIONS =====
def distancia_perpendicular(A, B, PC):
    (xA, yA), (xB, yB), (xPC, yPC) = A, B, PC
    det = (xB - xA)*(yA - yPC) - (yB - yA)*(xA - xPC)
    AB = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    if AB == 0:
        return float('inf')
    d = -det / AB
    return d

def proyeccion(A, B, PC):
    A = np.array(A)
    B = np.array(B)
    PC = np.array(PC)
    AB = B - A
    AP = PC - A
    dot_product = np.dot(AP, AB)
    if np.dot(AB, AB) == 0:
        return A
    t = dot_product / np.dot(AB, AB)
    return A + t*AB

def calcular_distancia(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def dividir_segmento(A, B, num_partes):
    A = np.array(A)
    B = np.array(B)
    puntos = []
    for i in range(num_partes + 1):
        t = i / num_partes
        punto = A + t * (B - A)
        puntos.append((float(punto[0]), float(punto[1])))
    return puntos

# ===== TAB 2 FUNCTIONS =====
def calculate_polygon_area(coordinates):
    if len(coordinates) < 3:
        return 0.0
    n = len(coordinates)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i][0] * coordinates[j][1]
        area -= coordinates[i][1] * coordinates[j][0]
    return abs(area) / 2.0

def azimuth_to_coordinates(azimuth, distance, ref_x=0.0, ref_y=0.0):
    azimuth_rad = math.radians(azimuth)
    x_offset = math.sin(azimuth_rad) * distance
    y_offset = distance * math.cos(azimuth_rad)
    x = ref_x + x_offset
    y = ref_y + y_offset
    return round(x, 3), round(y, 3)

def parse_dms_to_decimal(dms_string):
    try:
        dms_string = str(dms_string).strip().replace(',', '.')
        patterns = [
            r'(\d+(?:\.\d+)?)[°d]\s*(\d+(?:\.\d+)?)[\'m]\s*(\d+(?:\.\d+)?)[\"\'s]?',
            r'^(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)$',
            r'^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$',
            r'^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$',
            r'^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$'
        ]
        for pattern in patterns:
            match = re.search(pattern, dms_string)
            if match and len(match.groups()) == 3:
                degrees = float(match.group(1))
                minutes = float(match.group(2))
                seconds = float(match.group(3))
                decimal_degrees = (((seconds / 60.0) + minutes) / 60.0) + degrees
                return decimal_degrees
        return float(dms_string.replace(',', '.'))
    except (ValueError, AttributeError):
        return None

def validate_azimuth(azimuth):
    return 0 <= azimuth <= 360

# ===== MAIN APP =====
def main():
    # Language selector in sidebar
    st.sidebar.header("🌍 Configuración / Settings")
    lang = st.sidebar.radio("Idioma / Language", ['Español', 'English'], index=0)
    lang_code = 'es' if lang == 'Español' else 'en'
    
    # Main tabs
    tab1, tab2 = st.tabs([get_text('tab1', lang_code), get_text('tab2', lang_code)])
    
    with tab1:
        st.markdown(f'<div class="main-header">{get_text("title_tab1", lang_code)}</div>', unsafe_allow_html=True)
        st.markdown(get_text('subtitle_tab1', lang_code))
        
        # Sidebar inputs for Tab 1
        with st.sidebar:
            st.header("🔧 Parámetros de Entrada")
            
            st.subheader("Coordenadas del Punto A")
            xA = float(st.text_input("Coordenada X A", value="1072.998", key="xA"))
            yA = float(st.text_input("Coordenada Y A", value="971.948", key="yA"))
            
            st.subheader("Coordenadas del Punto B")
            xB = float(st.text_input("Coordenada X B", value="963.595", key="xB"))
            yB = float(st.text_input("Coordenada Y B", value="1012.893", key="yB"))
            
            st.subheader("Coordenadas del Punto PC")
            xPC = float(st.text_input("Coordenada X PC", value="1040.749", key="xPC"))
            yPC = float(st.text_input("Coordenada Y PC", value="983.875", key="yPC"))
            
            tol = st.slider("Tolerancia (m)", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
            
            st.subheader("🔢 División del Segmento AB")
            num_divisions = st.number_input("Número de divisiones", min_value=1, max_value=20, value=5, step=1)
        
        # Calculations for Tab 1
        A = (xA, yA)
        B = (xB, yB)
        PC = (xPC, yPC)
        
        if calcular_distancia(A, B) < 0.001:
            st.error("❌ Los puntos A y B son demasiado cercanos o iguales.")
            st.stop()
        
        d_signed = distancia_perpendicular(A, B, PC)
        d_abs = abs(d_signed)
        proj = proyeccion(A, B, PC)
        corr_vector = proj - np.array(PC)
        alineado = d_abs <= tol
        dist_perp = calcular_distancia(PC, proj)
        dist_AB = calcular_distancia(A, B)
        puntos_division = dividir_segmento(A, B, num_divisions)
        longitud_entre_puntos = dist_AB / num_divisions
        
        # Display results for Tab 1
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Resultados de Alineación")
            
            with st.container():
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.metric("Distancia perpendicular absoluta", f"{d_abs:.3f} m", delta=f"{d_signed:.3f} m")
                st.metric("Distancia del segmento AB", f"{dist_AB:.3f} m")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if alineado:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success(f"✅ **PC está ALINEADO** con AB (dentro de la tolerancia de {tol} m)")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(f"⚠️ **PC NO está alineado** con AB (fuera de tolerancia de {tol} m)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if d_signed > 0:
                st.info("📍 **PC está a la DERECHA** de la línea AB")
            elif d_signed < 0:
                st.info("📍 **PC está a la IZQUIERDA** de la línea AB")
            else:
                st.info("🎯 **PC está exactamente sobre** la línea AB")
            
            st.subheader("📐 Detalles de Proyección")
            st.write(f"**Coordenadas de proyección:** ({proj[0]:.3f}, {proj[1]:.3f})")
            st.write(f"**Vector de corrección:** ΔX = {corr_vector[0]:.3f} m, ΔY = {corr_vector[1]:.3f} m")
            
            st.subheader("📏 División del Segmento AB")
            st.markdown('<div class="division-box">', unsafe_allow_html=True)
            st.write(f"**Segmento AB dividido en {num_divisions} partes iguales**")
            st.write(f"**Longitud entre puntos:** {longitud_entre_puntos:.3f} m")
            st.markdown('</div>', unsafe_allow_html=True)
            
            division_data = []
            for i, punto in enumerate(puntos_division):
                distancia_desde_A = calcular_distancia(A, punto)
                division_data.append({
                    "Punto": f"P{i}", "X": f"{punto[0]:.3f}", "Y": f"{punto[1]:.3f}",
                    "Distancia desde A": f"{distancia_desde_A:.3f} m"
                })
            
            st.table(division_data[:6])
            if len(division_data) > 6:
                with st.expander("Ver todos los puntos"):
                    for i in range(6, len(division_data)):
                        st.write(f"{division_data[i]['Punto']}: X={division_data[i]['X']}, Y={division_data[i]['Y']}")
        
        with col2:
            st.subheader("📈 Visualización Gráfica")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            ax.plot([xA, xB], [yA, yB], 'b-', linewidth=3, label="Línea AB", alpha=0.7)
            ax.plot([xPC, proj[0]], [yPC, proj[1]], 'r--', linewidth=2, label="Distancia perpendicular", alpha=0.7)
            
            division_x = [p[0] for p in puntos_division]
            division_y = [p[1] for p in puntos_division]
            ax.scatter(division_x, division_y, c='orange', s=50, alpha=0.7, label=f"Puntos de división ({num_divisions} partes)")
            
            for i, (x, y) in enumerate(puntos_division):
                if i == 0:
                    ax.text(x, y, '  A', verticalalignment='center', fontweight='bold', fontsize=10)
                elif i == len(puntos_division) - 1:
                    ax.text(x, y, '  B', verticalalignment='center', fontweight='bold', fontsize=10)
                else:
                    ax.text(x, y, f'  P{i}', verticalalignment='center', fontsize=8, alpha=0.8)
            
            ax.plot(xA, yA, 'bo', markersize=8)
            ax.plot(xB, yB, 'bo', markersize=8)
            
            mid_x = (xPC + proj[0]) / 2
            mid_y = (yPC + proj[1]) / 2
            ax.annotate('', xy=(proj[0], proj[1]), xytext=(xPC, yPC),
                        arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
            
            ax.plot(xPC, yPC, 'ro', markersize=12, markerfacecolor='red', label="Punto de Control (PC)")
            ax.text(xPC, yPC, '  PC', verticalalignment='center', fontweight='bold', color='red')
            
            ax.plot(proj[0], proj[1], 'gs', markersize=10, label="Proyección")
            ax.text(proj[0], proj[1], '  Proy', verticalalignment='center', fontweight='bold', color='green')
            
            offset_x, offset_y = 6, 6
            ax.text(mid_x + offset_x, mid_y + offset_y, f'd = {dist_perp:.3f} m',
                    backgroundcolor='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            margin = max(dist_perp, dist_AB * 0.1) + 2
            min_x = min(xA, xB, xPC, proj[0]) - margin
            max_x = max(xA, xB, xPC, proj[0]) + margin
            min_y = min(yA, yB, yPC, proj[1]) - margin
            max_y = max(yA, yB, yPC, proj[1]) + margin
            
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_xlabel("Coordenada X (m)")
            ax.set_ylabel("Coordenada Y (m)")
            ax.set_title(f"Visualización de Alineación PC-AB + División en {num_divisions} Partes", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.legend()
            st.pyplot(fig)
    
    with tab2:
        st.markdown(f'<div class="main-header">{get_text("title_tab2", lang_code)}</div>', unsafe_allow_html=True)
        st.markdown(get_text('subtitle_tab2', lang_code))
        
        # Sidebar inputs for Tab 2
        with st.sidebar:
            st.subheader(get_text('reference_point', lang_code))
            ref_x_tab2 = st.number_input(get_text('reference_x', lang_code), value=1000.0, key="ref_x_tab2")
            ref_y_tab2 = st.number_input(get_text('reference_y', lang_code), value=1000.0, key="ref_y_tab2")
        
        # Tab 2 content
        tab2_col1, tab2_col2 = st.columns(2)
        
        with tab2_col1:
            input_method = st.radio("Formato de entrada de azimut", ["GMS (266°56'7.24\")", "Decimal (266.935)"], horizontal=True)
            
            if input_method.startswith("GMS"):
                azimuth_input = st.text_input("Azimut (Formato fácil)", value="", placeholder="26 56 7.00 o 26-56-7.00 o 26:56:7.00")
                if azimuth_input:
                    azimuth = parse_dms_to_decimal(azimuth_input)
                    if azimuth is None:
                        st.error(f"❌ No se pudo analizar '{azimuth_input}'. Intenta formato como: 45°30'15\"")
                        azimuth = 0.0
                    else:
                        st.success(f"✅ Analizado: {azimuth_input} → {azimuth:.8f}°")
                        if not validate_azimuth(azimuth):
                            st.warning(f"⚠️ Azimut {azimuth:.3f}° está fuera del rango 0-360°")
                else:
                    azimuth = 0.0
                    st.info("👆 Ingresa un valor de azimut arriba")
            else:
                azimuth = st.number_input("Azimut (grados decimales)", min_value=0.0, max_value=360.0, value=0.0, step=0.001, format="%.3f")
            
            distance_tab2 = st.number_input(get_text('distance', lang_code), min_value=0.0, value=1.0, step=0.001, format="%.3f")
        
        with tab2_col2:
            if azimuth > 0 or distance_tab2 > 0:
                try:
                    x, y = azimuth_to_coordinates(azimuth, distance_tab2, ref_x_tab2, ref_y_tab2)
                    st.subheader(get_text('results', lang_code))
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.metric(get_text('x_coordinate', lang_code), f"{x:.3f}")
                    with col_y:
                        st.metric(get_text('y_coordinate', lang_code), f"{y:.3f}")
                    st.write(f"**Entrada:** Azimut {azimuth:.3f}°, Distancia {distance_tab2}, Punto de Referencia ({ref_x_tab2}, {ref_y_tab2})")
                except Exception as e:
                    st.error(f"❌ Error de cálculo: {str(e)}")
            else:
                st.info("👈 Ingresa valores de azimut y distancia para ver resultados")
        
        # Batch conversion section
        st.subheader("📊 Conversión por Lotes")
        st.markdown("Convierte múltiples puntos con recorrido de polígono (cada punto se convierte en referencia para el siguiente).")
        
        if 'batch_data' not in st.session_state:
            st.session_state.batch_data = pd.DataFrame({'Azimuth': [], 'Distance': []})
        
        input_method_batch = st.radio("Método de entrada", ["Entrada Manual", "Subir CSV"], horizontal=True, key="batch_method")
        
        if input_method_batch == "Entrada Manual":
            if not st.session_state.batch_data.empty:
                st.write("**Datos actuales:**")
                st.dataframe(st.session_state.batch_data, use_container_width=True)
            
            if 'form_counter' not in st.session_state:
                st.session_state.form_counter = 0
                
            with st.form(f"add_entry_form_{st.session_state.form_counter}"):
                st.write("**Agregar nueva entrada:**")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    new_azimuth = st.text_input("Azimut", value="", placeholder="26 56 7.00 o 26.935")
                with col2:
                    new_distance = st.number_input("Distancia", value=None, step=0.001, format="%.3f")
                with col3:
                    submitted = st.form_submit_button("➕ Agregar")
                    
                if submitted and new_azimuth and new_distance is not None and new_distance > 0:
                    new_row = pd.DataFrame({'Azimuth': [new_azimuth], 'Distance': [new_distance]})
                    st.session_state.batch_data = pd.concat([st.session_state.batch_data, new_row], ignore_index=True)
                    st.session_state.form_counter += 1
                    st.success("✅ Entrada agregada!")
                    st.rerun()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Limpiar datos"):
                    st.session_state.batch_data = pd.DataFrame({'Azimuth': [], 'Distance': []})
                    st.rerun()
            with col2:
                if st.button("📝 Ejemplos"):
                    st.session_state.batch_data = pd.DataFrame({
                        'Azimuth': ["26 56 7.00", "90-0-0", "180:30:15.5", "270_45_30"],
                        'Distance': [5.178, 1.000, 1.000, 1.000]
                    })
                    st.rerun()
        
        else:
            uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    if 'Azimuth' in uploaded_df.columns and 'Distance' in uploaded_df.columns:
                        st.session_state.batch_data = uploaded_df[['Azimuth', 'Distance']]
                        st.success("✅ Archivo subido exitosamente!")
                        st.dataframe(st.session_state.batch_data)
                    else:
                        st.error("❌ CSV debe contener columnas 'Azimuth' y 'Distance'")
                except Exception as e:
                    st.error(f"❌ Error leyendo archivo: {str(e)}")
        
        if st.button("🔄 Convertir Todo", type="primary"):
            if not st.session_state.batch_data.empty:
                results = []
                errors = []
                current_ref_x = ref_x_tab2
                current_ref_y = ref_y_tab2
                
                for index, row in st.session_state.batch_data.iterrows():
                    try:
                        azimuth_raw = row['Azimuth']
                        if isinstance(azimuth_raw, str):
                            azimuth_val = parse_dms_to_decimal(azimuth_raw)
                            if azimuth_val is None:
                                errors.append(f"Fila {int(index) + 1}: Formato de azimut inválido '{azimuth_raw}'")
                                continue
                        else:
                            azimuth_val = float(azimuth_raw)
                        
                        distance_val = float(row['Distance'])
                        
                        if not validate_azimuth(azimuth_val):
                            errors.append(f"Fila {int(index) + 1}: Azimut inválido {azimuth_val}°")
                            continue
                        
                        x, y = azimuth_to_coordinates(azimuth_val, distance_val, current_ref_x, current_ref_y)
                        
                        results.append({
                            'Fila': int(index) + 1,
                            'Azimut_Original': str(azimuth_raw),
                            'Azimut_Decimal': float(azimuth_val),
                            'Distancia': float(distance_val),
                            'Referencia_X': float(current_ref_x),
                            'Referencia_Y': float(current_ref_y),
                            'Coordenada_X': float(x),
                            'Coordenada_Y': float(y)
                        })
                        
                        current_ref_x, current_ref_y = x, y
                        
                    except Exception as e:
                        errors.append(f"Fila {int(index) + 1}: {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.success(f"✅ Convertidos exitosamente {len(results)} puntos")
                    
                    final_x, final_y = results_df.iloc[-1]['Coordenada_X'], results_df.iloc[-1]['Coordenada_Y']
                    closure_error = math.sqrt((final_x - ref_x_tab2)**2 + (final_y - ref_y_tab2)**2)
                    
                    if closure_error < 0.01:
                        st.success(f"🎯 Polígono CERRADO! Error: {closure_error:.6f}")
                    else:
                        st.error(f"⚠️ Error de cierre del polígono: {closure_error:.6f}")
                    
                    coordinates = [(ref_x_tab2, ref_y_tab2)]
                    for _, row in results_df.iterrows():
                        coordinates.append((row['Coordenada_X'], row['Coordenada_Y']))
                    
                    polygon_area = calculate_polygon_area(coordinates)
                    st.subheader("📐 Área del Polígono")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Área", f"{polygon_area:.3f} m²")
                    with col2:
                        st.metric("Vértices", f"{len(results)}")
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Descargar Resultados CSV",
                        data=csv_buffer.getvalue(),
                        file_name="resultados_azimut.csv",
                        mime="text/csv"
                    )
                
                if errors:
                    st.error("❌ Errores encontrados:")
                    for error in errors:
                        st.write(f"- {error}")
            else:
                st.warning("⚠️ No hay datos para convertir")

if __name__ == "__main__":
    main()
