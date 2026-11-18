"""
Prototipo: Random Forest aplicado al rendimiento agrícola — 3 escenarios
Curso: Modelamiento predictivo de datos (Universidad de Lima) — Grupo 3, técnica Random Forest

Ejecutar (sin internet):
    pip install -r requirements.txt
    streamlit run app_rf_escenarios.py

El archivo agriculture_dataset_synthetic.csv debe estar en la misma carpeta
(o se puede subir desde la barra lateral).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Random Forest – Escenarios", layout="wide", page_icon="🌳")

VERDE = "#2C5F2D"
VERDE_CLARO = "#97BC62"
ROJO = "#B23A48"

# ======================================================================
# BLOQUE 1. CARGA Y PREPARACIÓN DE DATOS
# ======================================================================
RENOMBRAR = {
    "Farm_Area(acres)": "Farm_Area",
    "Fertilizer_Used(tons)": "Fertilizer",
    "Pesticide_Used(kg)": "Pesticide",
    "Yield(tons)": "Yield",
    "Water_Usage(cubic meters)": "Water_Usage",
}
NUM = ["Farm_Area", "Fertilizer", "Pesticide", "Water_Usage"]
CAT = ["Crop_Type", "Irrigation_Type", "Soil_Type", "Season"]

st.sidebar.header("1. Datos")
archivo = st.sidebar.file_uploader("Sube el CSV (opcional)", type=["csv"])
RUTA_DEFECTO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agriculture_dataset_synthetic.csv")  # el CSV está en la carpeta raíz de la app

if archivo is not None:
    df = pd.read_csv(archivo)
elif os.path.exists(RUTA_DEFECTO):
    df = pd.read_csv(RUTA_DEFECTO)
else:
    st.warning("Coloca agriculture_dataset_synthetic.csv en la carpeta de la app o súbelo desde la barra lateral.")
    st.stop()

df = df.rename(columns=RENOMBRAR)
# Indicador derivado: rendimiento por acre (quita el efecto del tamaño de la finca)
df["Yield_per_acre"] = df["Yield"] / df["Farm_Area"]

# One-hot encoding de las variables categóricas (cada categoría pasa a ser una columna 0/1)
X_full = pd.get_dummies(df[NUM + CAT], columns=CAT).astype(float)

# ======================================================================
# BLOQUE 2. HIPERPARÁMETROS DEL RANDOM FOREST (iguales para los 3 escenarios)
# ======================================================================
st.sidebar.header("2. Hiperparámetros del Random Forest")
n_arboles = st.sidebar.slider("n_estimators (número de árboles)", 100, 800, 500, 100)
prof_opc = st.sidebar.selectbox("max_depth (profundidad máxima)", ["Sin límite", 5, 10, 15], index=0)
max_depth = None if prof_opc == "Sin límite" else int(prof_opc)
min_hoja = st.sidebar.slider("min_samples_leaf (mínimo de fincas por hoja)", 1, 10, 1)
test_size = st.sidebar.slider("Tamaño del conjunto de prueba", 0.1, 0.4, 0.2, 0.05)
SEMILLA = 42

# El MISMO split para los 3 escenarios -> la comparación es justa
idx_train, idx_test = train_test_split(df.index, test_size=test_size, random_state=SEMILLA)


def nuevo_rf():
    return RandomForestRegressor(
        n_estimators=n_arboles, max_depth=max_depth, min_samples_leaf=min_hoja,
        random_state=SEMILLA, n_jobs=-1,
    )


def metricas(y_real, y_pred):
    return {
        "R²": r2_score(y_real, y_pred),
        "RMSE (t)": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAE (t)": mean_absolute_error(y_real, y_pred),
    }


# ======================================================================
# BLOQUE 3. LOS TRES ESCENARIOS
# ======================================================================
@st.cache_resource(show_spinner="Entrenando los 3 escenarios…")
def entrenar_escenarios(_df, _X, n_arboles, max_depth, min_hoja, test_size, metodo_red):
    y = _df["Yield"]
    tr, te = idx_train, idx_test
    res = {}

    # ---- Escenario 1: modelo base (como en la EC1), todas las variables, target = Yield total
    rf1 = nuevo_rf().fit(_X.loc[tr], y[tr])
    p1 = rf1.predict(_X.loc[te])
    res["E1"] = dict(modelo=rf1, cols=list(_X.columns), pred=p1, pred_train=rf1.predict(_X.loc[tr]),
                     imp=pd.Series(rf1.feature_importances_, _X.columns))

    # ---- Escenario 2: reducción de dimensiones
    if metodo_red == "Eliminar Water_Usage (VIF > 10)":
        cols2 = [c for c in _X.columns if c != "Water_Usage"]
        X2_tr, X2_te = _X.loc[tr, cols2], _X.loc[te, cols2]
    else:  # PCA sobre las 4 numéricas estandarizadas -> 3 componentes
        sc = StandardScaler().fit(_X.loc[tr, NUM])
        pca = PCA(n_components=3).fit(sc.transform(_X.loc[tr, NUM]))

        def transformar(D):
            comp = pd.DataFrame(pca.transform(sc.transform(D[NUM])), columns=["PC1", "PC2", "PC3"], index=D.index)
            return pd.concat([comp, D.drop(columns=NUM)], axis=1)

        X2_tr, X2_te = transformar(_X.loc[tr]), transformar(_X.loc[te])
        cols2 = list(X2_tr.columns)
    rf2 = nuevo_rf().fit(X2_tr, y[tr])
    p2 = rf2.predict(X2_te)
    res["E2"] = dict(modelo=rf2, cols=cols2, pred=p2, pred_train=rf2.predict(X2_tr),
                     imp=pd.Series(rf2.feature_importances_, cols2))

    # ---- Escenario 3: el RF predice rendimiento POR ACRE con variables de manejo;
    #      luego Yield total = predicción por acre × área de la finca
    cols3 = [c for c in _X.columns if c not in ("Farm_Area", "Water_Usage")]
    ya = _df["Yield_per_acre"]
    rf3 = nuevo_rf().fit(_X.loc[tr, cols3], ya[tr])
    pa_te = rf3.predict(_X.loc[te, cols3])
    pa_tr = rf3.predict(_X.loc[tr, cols3])
    res["E3"] = dict(modelo=rf3, cols=cols3,
                     pred=pa_te * _df.loc[te, "Farm_Area"].values,
                     pred_train=pa_tr * _df.loc[tr, "Farm_Area"].values,
                     pred_acre=pa_te, r2_acre=r2_score(ya[te], pa_te),
                     imp=pd.Series(rf3.feature_importances_, cols3))
    return res


metodo_red = st.sidebar.radio("Escenario 2: método de reducción de dimensiones",
                              ["Eliminar Water_Usage (VIF > 10)", "PCA (3 componentes)"])

RES = entrenar_escenarios(df, X_full, n_arboles, max_depth, min_hoja, test_size, metodo_red)
y_test = df.loc[idx_test, "Yield"]
y_train = df.loc[idx_train, "Yield"]

NOMBRES = {
    "E1": "Escenario 1 · Base (todas las variables)",
    "E2": "Escenario 2 · Reducción de dimensiones",
    "E3": "Escenario 3 · Rendimiento por acre × área",
}

tabla = []
for k in ["E1", "E2", "E3"]:
    m_te = metricas(y_test, RES[k]["pred"])
    m_tr = r2_score(y_train, RES[k]["pred_train"])
    tabla.append({"Escenario": NOMBRES[k], "N.° variables": len(RES[k]["cols"]), **m_te,
                  "R² entrenamiento": m_tr, "Brecha R² (train − test)": m_tr - m_te["R²"]})
tabla = pd.DataFrame(tabla).set_index("Escenario")

# ======================================================================
# INTERFAZ
# ======================================================================
# Encabezado con el nombre del alumno (para la captura que pide el examen)
nombre = st.sidebar.text_input("Nombre para el encabezado", "Grupo 3 · Bajalqui De la Cruz, L. M. · De La Cuba Oliva, F. A. · Perdicci Taboada, G. G.")
st.markdown(f"<div style='background:#2C5F2D;color:white;padding:10px 16px;border-radius:8px'>"
            f"<b>{nombre}</b> · Modelamiento predictivo de datos · Random Forest – 3 escenarios</div>",
            unsafe_allow_html=True)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # para importar estilo.py
from estilo import hero, guia  # noqa: E402
hero("🌳 Random Forest: tres escenarios y simulador",
     "Cientos de árboles de decisión votan y se promedia su respuesta. Aquí comparamos tres formas de usarlo y "
     "simulamos qué pasa con el rendimiento si cambia una decisión de manejo.", img="bosque")
guia(["**1️⃣ Contexto:** mire qué variables se relacionan más con el rendimiento (correlación) y cuáles se repiten (VIF).",
      "**2️⃣ Tres escenarios:** compare el modelo con todas las variables, sin la variable redundante y con PCA. "
      "Barra más larga en R² = mejor.",
      "**3️⃣ Simulador:** elija cultivo, riego, suelo, temporada e insumos y vea la predicción; abajo verá el efecto "
      "promedio en las 600 fincas.",
      "**4️⃣ Conclusiones:** qué decisión de gestión se desprende del modelo.",
      "En la barra lateral puede cambiar el número de árboles y la profundidad para ver cómo cambia la precisión."])
st.caption("Dimensión 1 de la EC1 (Yield) · Dataset sintético de 600 explotaciones · Mismo split 80/20 "
           f"(random_state = {SEMILLA}) en todos los escenarios")

tab_ctx, tab_esc, tab_sim, tab_concl = st.tabs(
    ["1️⃣ Contexto y correlación", "2️⃣ Tres escenarios", "3️⃣ Simulador ¿qué pasa si?", "4️⃣ Conclusiones de gestión"]
)

# ----------------------------------------------------------------------
# TAB 1: sustento correlacional
# ----------------------------------------------------------------------
with tab_ctx:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Explotaciones", f"{len(df):,}")
    c2.metric("Yield promedio", f"{df['Yield'].mean():.2f} t")
    c3.metric("Yield por acre promedio", f"{df['Yield_per_acre'].mean():.3f} t/acre")
    c4.metric("Corr. Farm_Area ↔ Yield", f"{df['Farm_Area'].corr(df['Yield']):.3f}")

    st.subheader("Correlación de Pearson con el rendimiento")
    corr = df[NUM + ["Yield", "Yield_per_acre"]].corr()[["Yield", "Yield_per_acre"]].drop(["Yield", "Yield_per_acre"])
    col_a, col_b = st.columns([1, 1.2])
    with col_a:
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)
        st.caption("El rendimiento **total** se explica casi solo por el tamaño (Farm_Area). "
                   "Al pasar a rendimiento **por acre**, el área deja de dominar y aparece el aporte "
                   "del manejo (fertilizante, riego, cultivo).")
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        corr.plot(kind="barh", ax=ax, color=[VERDE, VERDE_CLARO])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Correlación de Pearson")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    st.subheader("Multicolinealidad (VIF) de las variables numéricas")
    vif = []
    for col in NUM:
        otras = [c for c in NUM if c != col]
        r2v = LinearRegression().fit(df[otras], df[col]).score(df[otras], df[col])
        vif.append({"Variable": col, "VIF": 1 / (1 - r2v)})
    vif = pd.DataFrame(vif).set_index("Variable")
    st.dataframe(vif.style.format("{:.2f}"), use_container_width=True)
    st.caption("Farm_Area y Water_Usage tienen VIF > 10: aportan casi la misma información. "
               "Esto justifica el **Escenario 2** (reducción de dimensiones).")

    st.subheader("Rendimiento por acre según tipo de riego y cultivo")
    col_r, col_c = st.columns(2)
    with col_r:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        df.groupby("Irrigation_Type")["Yield_per_acre"].mean().sort_values().plot(kind="barh", ax=ax, color=VERDE_CLARO)
        ax.set_xlabel("t/acre promedio")
        st.pyplot(fig)
    with col_c:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        df.groupby("Crop_Type")["Yield_per_acre"].mean().sort_values().plot(kind="barh", ax=ax, color=VERDE)
        ax.set_xlabel("t/acre promedio")
        st.pyplot(fig)

# ----------------------------------------------------------------------
# TAB 2: los tres escenarios
# ----------------------------------------------------------------------
with tab_esc:
    st.subheader("Comparación de los tres escenarios (conjunto de prueba)")
    st.dataframe(tabla.style.format("{:.3f}", subset=["R²", "R² entrenamiento", "Brecha R² (train − test)"])
                 .format("{:.2f}", subset=["RMSE (t)", "MAE (t)"]), use_container_width=True)
    st.caption("Referencia del informe EC1 (Tabla 4.1): Regresión lineal R² = 0.919 · Random Forest R² = 0.829 · "
               "XGBoost R² = 0.947. La brecha train − test grande indica sobreajuste.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    tabla["R²"].plot(kind="bar", ax=axes[0], color=[VERDE_CLARO, VERDE_CLARO, VERDE])
    axes[0].axhline(0.947, color=ROJO, linestyle="--", linewidth=1, label="XGBoost EC1 (0.947)")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("R² en prueba")
    axes[0].set_xticklabels(["E1", "E2", "E3"], rotation=0)
    axes[0].legend(fontsize=8)
    tabla["RMSE (t)"].plot(kind="bar", ax=axes[1], color=[VERDE_CLARO, VERDE_CLARO, VERDE])
    axes[1].set_title("RMSE en prueba (t) — menor es mejor")
    axes[1].set_xticklabels(["E1", "E2", "E3"], rotation=0)
    fig.tight_layout()
    st.pyplot(fig)

    DESCRIPCION = {
        "E1": "Réplica del modelo de la EC1: las 8 variables originales (área, insumos, agua y categóricas) "
              "para predecir el rendimiento total. Sirve como línea base.",
        "E2": "Se aplica reducción de dimensiones porque Farm_Area y Water_Usage son redundantes (VIF > 10). "
              "Se comprueba si el modelo mantiene su precisión con menos información repetida.",
        "E3": "Se separa el efecto escala del efecto manejo: el Random Forest aprende el rendimiento POR ACRE "
              "a partir de cultivo, riego, suelo, temporada e insumos; luego se multiplica por el área.",
    }
    for k in ["E1", "E2", "E3"]:
        with st.expander(NOMBRES[k], expanded=(k == "E3")):
            st.write(DESCRIPCION[k])
            m = tabla.loc[NOMBRES[k]]
            a, b, c, d = st.columns(4)
            a.metric("R² prueba", f"{m['R²']:.3f}")
            b.metric("RMSE", f"{m['RMSE (t)']:.2f} t")
            c.metric("MAE", f"{m['MAE (t)']:.2f} t")
            d.metric("N.° variables", int(m["N.° variables"]))
            if k == "E3":
                st.caption(f"R² del modelo por acre (antes de multiplicar por el área): {RES['E3']['r2_acre']:.3f}")

            g1, g2 = st.columns(2)
            with g1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, RES[k]["pred"], alpha=0.6, color=VERDE)
                lim = [0, max(y_test.max(), RES[k]["pred"].max())]
                ax.plot(lim, lim, "r--", linewidth=1)
                ax.set_xlabel("Yield real (t)")
                ax.set_ylabel("Yield predicho (t)")
                ax.set_title("Predicho vs. real")
                st.pyplot(fig)
            with g2:
                fig, ax = plt.subplots(figsize=(5, 4))
                RES[k]["imp"].sort_values(ascending=False).head(10).sort_values().plot(kind="barh", ax=ax, color=VERDE_CLARO)
                ax.set_title("Importancia de variables (top 10)")
                st.pyplot(fig)

# ----------------------------------------------------------------------
# TAB 3: simulador "qué pasa si" (sustento causal basado en el modelo)
# ----------------------------------------------------------------------
with tab_sim:
    st.subheader("¿Qué pasa con el rendimiento si cambio una decisión de manejo?")
    st.caption("Se usa el modelo del Escenario 3. Se mantiene todo igual y se cambia UNA variable: así se aísla "
               "su efecto sobre la predicción (análisis contrafactual con el modelo).")

    rf3, cols3 = RES["E3"]["modelo"], RES["E3"]["cols"]

    def fila_modelo(cultivo, riego, suelo, temporada, fert, pest):
        fila = pd.DataFrame(0.0, index=[0], columns=cols3)
        fila["Fertilizer"], fila["Pesticide"] = fert, pest
        for col, val in [("Crop_Type", cultivo), ("Irrigation_Type", riego), ("Soil_Type", suelo), ("Season", temporada)]:
            nombre = f"{col}_{val}"
            if nombre in fila.columns:
                fila[nombre] = 1.0
        return fila

    s1, s2, s3 = st.columns(3)
    with s1:
        cultivo = st.selectbox("Cultivo", sorted(df["Crop_Type"].unique()))
        suelo = st.selectbox("Suelo", sorted(df["Soil_Type"].unique()))
    with s2:
        riego = st.selectbox("Tipo de riego", sorted(df["Irrigation_Type"].unique()))
        temporada = st.selectbox("Temporada", sorted(df["Season"].unique()))
    with s3:
        area = st.number_input("Área (acres)", 5.0, 400.0, float(round(df["Farm_Area"].median(), 1)), 5.0)
        fert = st.slider("Fertilizante (t)", 0.5, 10.0, float(round(df["Fertilizer"].mean(), 1)), 0.1)
        pest = st.slider("Pesticida (kg)", 0.2, 5.0, float(round(df["Pesticide"].mean(), 1)), 0.1)

    por_acre = rf3.predict(fila_modelo(cultivo, riego, suelo, temporada, fert, pest))[0]
    r1, r2 = st.columns(2)
    r1.metric("Rendimiento por acre predicho", f"{por_acre:.3f} t/acre")
    r2.metric("Rendimiento total predicho", f"{por_acre * area:.1f} t")

    st.markdown("**Misma finca, cambiando solo el tipo de riego**")
    filas = []
    for alt in sorted(df["Irrigation_Type"].unique()):
        pa = rf3.predict(fila_modelo(cultivo, alt, suelo, temporada, fert, pest))[0]
        filas.append({"Riego": alt, "t/acre": pa, "Yield total (t)": pa * area,
                      "Cambio vs. selección (t)": (pa - por_acre) * area})
    esc_riego = pd.DataFrame(filas).set_index("Riego").sort_values("t/acre", ascending=False)
    st.dataframe(esc_riego.style.format("{:.2f}"), use_container_width=True)

    st.divider()
    st.subheader("Efecto promedio en las 600 fincas (contrafactual)")
    st.caption("A todas las fincas se les asigna el mismo tipo de riego, manteniendo el resto de sus "
               "características reales. La diferencia frente a la producción actual estima el impacto de la decisión.")
    X3_todas = X_full[cols3].copy()
    riegos = [c for c in cols3 if c.startswith("Irrigation_Type_")]
    prod_actual = float((rf3.predict(X3_todas) * df["Farm_Area"]).sum())
    contraf = []
    for c in riegos:
        Z = X3_todas.copy()
        Z[riegos] = 0.0
        Z[c] = 1.0
        total = float((rf3.predict(Z) * df["Farm_Area"]).sum())
        contraf.append({"Todas con riego": c.replace("Irrigation_Type_", ""), "Producción total (t)": total,
                        "Variación vs. actual (%)": (total / prod_actual - 1) * 100})
    contraf = pd.DataFrame(contraf).set_index("Todas con riego").sort_values("Producción total (t)", ascending=False)
    k1, k2 = st.columns([1, 1.2])
    with k1:
        st.metric("Producción actual estimada", f"{prod_actual:,.0f} t")
        st.dataframe(contraf.style.format({"Producción total (t)": "{:,.0f}", "Variación vs. actual (%)": "{:+.1f}"}),
                     use_container_width=True)
    with k2:
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        variacion = contraf["Variación vs. actual (%)"].sort_values()
        colores = [VERDE if v > 0 else ROJO for v in variacion]
        ax.barh(variacion.index, variacion.values, color=colores)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Variación de la producción total (%)")
        st.pyplot(fig)

    st.markdown("**Escenarios pesimista / base / optimista (decisión de riego)**")
    esc_pbo = pd.DataFrame({
        "Supuesto": ["Todas las fincas con riego Rain-fed", "Situación actual", "Todas las fincas con riego Drip"],
        "Producción total (t)": [contraf.loc["Rain-fed", "Producción total (t)"], prod_actual,
                                 contraf.loc["Drip", "Producción total (t)"]],
    }, index=["Pesimista", "Base", "Optimista"])
    esc_pbo["Variación vs. base (%)"] = (esc_pbo["Producción total (t)"] / prod_actual - 1) * 100
    st.dataframe(esc_pbo.style.format({"Producción total (t)": "{:,.0f}", "Variación vs. base (%)": "{:+.1f}"}),
                 use_container_width=True)

    st.subheader("Curva de dependencia parcial: fertilizante")
    st.caption("Para cada nivel de fertilizante se fija ese valor en las 600 fincas y se promedia la predicción. "
               "Muestra cómo responde el rendimiento por acre cuando SOLO cambia el fertilizante.")
    niveles = np.linspace(df["Fertilizer"].min(), df["Fertilizer"].max(), 20)
    curva = []
    for v in niveles:
        Z = X3_todas.copy()
        Z["Fertilizer"] = v
        curva.append(rf3.predict(Z).mean())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(niveles, curva, marker="o", color=VERDE)
    ax.set_xlabel("Fertilizante (t)")
    ax.set_ylabel("t/acre promedio predicho")
    st.pyplot(fig)

# ----------------------------------------------------------------------
# TAB 4: conclusiones de gestión
# ----------------------------------------------------------------------
with tab_concl:
    e1, e3 = tabla.iloc[0], tabla.iloc[2]
    mejor_riego = contraf.index[0]
    peor_riego = contraf.index[-1]
    st.subheader("Aporte del Random Forest a los indicadores del caso")
    st.markdown(
        f"""
- **Escenario 1 (base):** R² = {e1['R²']:.3f}, RMSE = {e1['RMSE (t)']:.1f} t. El modelo depende casi por completo
  de Farm_Area (importancia ≈ {RES['E1']['imp'].max():.0%}): predice bien *porque las fincas grandes producen más*,
  pero eso no ayuda a decidir cómo gestionar una finca.
- **Escenario 2 (reducción de dimensiones):** con {int(tabla.iloc[1]['N.° variables'])} variables el R² es
  {tabla.iloc[1]['R²']:.3f}. Quitar la información redundante **no empeora** el modelo, lo hace más simple y
  estable.
- **Escenario 3 (por acre × área):** R² = {e3['R²']:.3f}, RMSE = {e3['RMSE (t)']:.1f} t. Es el mejor escenario:
  reduce el error en {(1 - e3['RMSE (t)'] / e1['RMSE (t)']):.0%} frente a la base y supera al XGBoost de la EC1
  (0.947). Además, las variables importantes pasan a ser de **manejo** (cultivo, riego, fertilizante), que sí son
  decisiones del productor.
- **Simulación:** con todo lo demás igual, pasar todas las fincas a riego **{mejor_riego}** cambia la producción
  total en {contraf.iloc[0]['Variación vs. actual (%)']:+.1f}%, mientras que **{peor_riego}** la cambia en
  {contraf.iloc[-1]['Variación vs. actual (%)']:+.1f}%.
        """
    )
    st.subheader("Decisiones que apoya")
    st.markdown(
        "- **Planificación de cosecha:** estimar las toneladas por finca con un error medio "
        f"de ±{e3['MAE (t)']:.1f} t.\n"
        "- **Inversión en riego:** priorizar la conversión a los sistemas con mayor efecto estimado.\n"
        "- **Selección de cultivo por campaña:** comparar t/acre esperados antes de sembrar."
    )
    st.info("Limitaciones: datos sintéticos, sin variables climáticas. La simulación es un contrafactual "
            "**según el modelo**; para afirmar causalidad real se necesitaría un ensayo de campo controlado.")
