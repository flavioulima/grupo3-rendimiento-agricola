"""
Prototipo: XGBoost en las 4 dimensiones del caso Farm Yield – escenarios Pesimista / Base / Optimista
Curso: Modelamiento predictivo de datos (Universidad de Lima) – Grupo 3 – Parte XGBoost

Ejecutar (funciona sin internet):
    pip install streamlit pandas numpy scikit-learn xgboost matplotlib
    streamlit run app_xgboost.py
El archivo agriculture_dataset_synthetic.csv debe estar en la misma carpeta.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="XGBoost – 4 dimensiones", layout="wide", page_icon="🌾")
VERDE, GRIS, ROJO, CLARO = "#2C5F2D", "#7F7F7F", "#B23A48", "#97BC62"
COLOR_ESC = {"Pesimista": ROJO, "Base": GRIS, "Optimista": VERDE}

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
RUTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agriculture_dataset_synthetic.csv")  # el CSV está en la carpeta raíz de la app
archivo = st.sidebar.file_uploader("CSV (opcional)", type=["csv"])
if archivo is not None:
    df = pd.read_csv(archivo)
elif os.path.exists(RUTA):
    df = pd.read_csv(RUTA)
else:
    st.warning("Coloca agriculture_dataset_synthetic.csv junto a la app o súbelo en la barra lateral.")
    st.stop()

df = df.rename(columns={"Farm_Area(acres)": "Farm_Area", "Fertilizer_Used(tons)": "Fertilizer",
                        "Pesticide_Used(kg)": "Pesticide", "Yield(tons)": "Yield",
                        "Water_Usage(cubic meters)": "Water_Usage"})
df["Yield_per_acre"] = df["Yield"] / df["Farm_Area"]
UMBRAL = df["Yield_per_acre"].median()
df["High_Productivity"] = (df["Yield_per_acre"] >= UMBRAL).astype(int)
CAT = ["Crop_Type", "Irrigation_Type", "Soil_Type", "Season"]

DIM = {
    "D1": dict(nombre="Dimensión 1 · Productividad agrícola", target="Yield", unidad="t",
               feats=["Farm_Area", "Crop_Type", "Irrigation_Type", "Fertilizer", "Pesticide", "Soil_Type", "Season", "Water_Usage"],
               num=["Farm_Area", "Fertilizer", "Pesticide", "Water_Usage"],
               params=dict(n_estimators=500, max_depth=3, learning_rate=0.03), informe="R² = 0.947 · RMSE = 17.26 t · MAE = 11.07 t"),
    "D2": dict(nombre="Dimensión 2 · Gestión de recursos hídricos", target="Water_Usage", unidad="m³",
               feats=["Farm_Area", "Crop_Type", "Irrigation_Type", "Soil_Type", "Season"], num=["Farm_Area"],
               params=dict(n_estimators=500, max_depth=3, learning_rate=0.03), informe="R² = 0.886 · RMSE = 8,114 m³ · MAE = 5,852 m³"),
    "D3": dict(nombre="Dimensión 3 · Uso de insumos agroquímicos", target="Yield_per_acre", unidad="t/acre",
               feats=["Fertilizer", "Pesticide", "Crop_Type", "Irrigation_Type", "Soil_Type", "Season"], num=["Fertilizer", "Pesticide"],
               params=dict(n_estimators=400, max_depth=3, learning_rate=0.05), informe="R² = 0.881 · RMSE = 0.0510 · MAE = 0.0387"),
    "D4": dict(nombre="Dimensión 4 · Cultivo y suelo (clasificación)", target="High_Productivity", unidad="",
               feats=["Crop_Type", "Soil_Type", "Season"], num=[],
               params=dict(n_estimators=300, max_depth=3, learning_rate=0.05), informe="AUC = 0.884 · Accuracy = 0.792 · F1 = 0.793"),
}

# ---------------------------------------------------------------------------
# 2. Hiperparámetros (opcional: el profe puede pedir cambiarlos)
# ---------------------------------------------------------------------------
st.sidebar.header("Hiperparámetros de XGBoost")
usar_informe = st.sidebar.checkbox("Usar los del informe", value=True)
if not usar_informe:
    n_est = st.sidebar.slider("n_estimators (rondas / árboles)", 50, 1000, 400, 50)
    prof = st.sidebar.slider("max_depth (profundidad de cada árbol)", 1, 8, 3)
    lr = st.sidebar.select_slider("learning_rate (tamaño del paso)", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.05)


def params_de(k):
    return DIM[k]["params"] if usar_informe else dict(n_estimators=n_est, max_depth=prof, learning_rate=lr)


# ---------------------------------------------------------------------------
# 3. Entrenamiento (80/20, random_state = 42)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando XGBoost…")
def entrenar(k, params_tuple):
    d = DIM[k]
    params = dict(params_tuple)
    X, y = df[d["feats"]], df[d["target"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if k == "D4" else None)
    categ = [c for c in d["feats"] if c in CAT]
    if k == "D4":
        prep = OneHotEncoder(handle_unknown="ignore")
        modelo = Pipeline([("prep", prep), ("xgb", XGBClassifier(random_state=42, eval_metric="logloss", **params))])
    else:
        prep = ColumnTransformer([("num", StandardScaler(), d["num"]), ("cat", OneHotEncoder(handle_unknown="ignore"), categ)])
        modelo = Pipeline([("prep", prep), ("xgb", XGBRegressor(random_state=42, eval_metric="rmse", **params))])
    # eval_set registra el error en cada ronda de boosting (curva de aprendizaje)
    prep_fit = modelo.named_steps["prep"].fit(Xtr)
    modelo.named_steps["xgb"].fit(prep_fit.transform(Xtr), ytr,
                                  eval_set=[(prep_fit.transform(Xtr), ytr), (prep_fit.transform(Xte), yte)], verbose=False)
    return modelo, Xtr, Xte, ytr, yte


def importancias(modelo, top=10):
    prep = modelo.named_steps["prep"]
    nombres = prep.get_feature_names_out() if hasattr(prep, "get_feature_names_out") else None
    imp = pd.Series(modelo.named_steps["xgb"].feature_importances_, index=nombres)
    imp.index = [str(i).replace("num__", "").replace("cat__", "") for i in imp.index]
    return imp.sort_values(ascending=False).head(top)


def grafico_barras_esc(tabla, col, titulo, fmt="{:,.0f}"):
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.bar(tabla.index, tabla[col], color=[COLOR_ESC[i] for i in tabla.index])
    for i, v in enumerate(tabla[col]):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(titulo)
    return fig


MODELOS = {k: entrenar(k, tuple(sorted(params_de(k).items()))) for k in DIM}

# ---------------------------------------------------------------------------
# 4. Interfaz
# ---------------------------------------------------------------------------
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # para importar estilo.py
from estilo import hero, guia  # noqa: E402
hero("🐆 XGBoost aplicado a las 4 dimensiones",
     "Árboles en fila: cada uno corrige el error del anterior. Lo aplicamos a la producción, el agua, los insumos y "
     "la probabilidad de productividad alta, cada uno con sus escenarios.", img="boost")
guia(["**📋 Resumen:** tabla con la precisión de XGBoost en cada dimensión frente al modelo base.",
      "**1️⃣–4️⃣:** en cada pestaña verá el gráfico real vs. predicho (puntos cerca de la diagonal = buen modelo), "
      "las variables más importantes, la curva de aprendizaje (cómo baja el error ronda a ronda) y un escenario de decisión.",
      "**🧠 ¿Cómo funciona XGBoost?:** el algoritmo explicado paso a paso.",
      "En la barra lateral, desmarque «Usar los del informe» para cambiar el número de árboles, la profundidad y la tasa de aprendizaje."])
st.caption(f"600 explotaciones · partición 80/20 · random_state = 42 · umbral de productividad alta = {UMBRAL:.4f} t/acre")

tabs = st.tabs(["📋 Resumen", "1️⃣ Productividad", "2️⃣ Agua", "3️⃣ Insumos", "4️⃣ Cultivo y suelo", "🧠 ¿Cómo funciona XGBoost?"])

# ---- Resumen
with tabs[0]:
    filas = []
    for k, (m, Xtr, Xte, ytr, yte) in MODELOS.items():
        if k == "D4":
            pr = m.predict_proba(Xte)[:, 1]
            filas.append({"Dimensión": DIM[k]["nombre"], "Métrica principal": f"AUC = {roc_auc_score(yte, pr):.3f}",
                          "Otras": f"Accuracy = {accuracy_score(yte, pr >= .5):.3f} · F1 = {f1_score(yte, pr >= .5):.3f}",
                          "Informe EC1": DIM[k]["informe"]})
        else:
            p = m.predict(Xte)
            filas.append({"Dimensión": DIM[k]["nombre"], "Métrica principal": f"R² = {r2_score(yte, p):.3f}",
                          "Otras": f"RMSE = {mean_squared_error(yte, p) ** .5:,.4g} · MAE = {mean_absolute_error(yte, p):,.4g} {DIM[k]['unidad']}",
                          "Informe EC1": DIM[k]["informe"]})
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.caption("Las pequeñas diferencias con el informe se deben a la versión de la librería xgboost; el método y los parámetros son los mismos.")
    st.markdown("""
**Escenarios de cada dimensión** (se cambia una decisión en las 600 fincas y el modelo vuelve a predecir):

| Dimensión | Pesimista | Base | Optimista |
|---|---|---|---|
| 1 · Productividad | Todas con riego Rain-fed | Situación actual | Todas con riego Drip |
| 2 · Agua | Todas Rain-fed | Situación actual | Todas Drip |
| 3 · Insumos | Fertilizante −20 %, pesticida +20 % | Situación actual | Fertilizante +20 %, pesticida −20 % |
| 4 · Cultivo y suelo | Peor temporada de siembra | Temporada actual | Mejor temporada de siembra |
""")


def bloque_modelo(k):
    m, Xtr, Xte, ytr, yte = MODELOS[k]
    d = DIM[k]
    c1, c2, c3 = st.columns(3)
    if k == "D4":
        pr = m.predict_proba(Xte)[:, 1]
        pred = (pr >= .5).astype(int)
        c1.metric("AUC-ROC", f"{roc_auc_score(yte, pr):.3f}")
        c2.metric("Accuracy", f"{accuracy_score(yte, pred):.3f}")
        c3.metric("F1", f"{f1_score(yte, pred):.3f}")
    else:
        p = m.predict(Xte)
        c1.metric("R² (prueba)", f"{r2_score(yte, p):.3f}", f"train {r2_score(ytr, m.predict(Xtr)):.3f}", delta_color="off")
        c2.metric("RMSE", f"{mean_squared_error(yte, p) ** .5:,.4g} {d['unidad']}")
        c3.metric("MAE", f"{mean_absolute_error(yte, p):,.4g} {d['unidad']}")
    st.caption(f"Informe EC1: {d['informe']} · Parámetros: {params_de(k)}")

    g1, g2, g3 = st.columns(3)
    with g1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        if k == "D4":
            fpr, tpr, _ = roc_curve(yte, pr)
            ax.plot(fpr, tpr, color=VERDE); ax.plot([0, 1], [0, 1], "r--", linewidth=1)
            ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos"); ax.set_title("Curva ROC")
        else:
            ax.scatter(yte, p, alpha=0.6, color=VERDE)
            lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]
            ax.plot(lim, lim, "r--", linewidth=1)
            ax.set_xlabel("Real"); ax.set_ylabel("Predicho"); ax.set_title("Predicho vs. real")
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        importancias(m).sort_values().plot(kind="barh", ax=ax, color=CLARO)
        ax.set_title("Importancia de variables")
        st.pyplot(fig)
    with g3:
        ev = m.named_steps["xgb"].evals_result()
        met = list(ev["validation_0"].keys())[0]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(ev["validation_0"][met], color=VERDE, label="Entrenamiento")
        ax.plot(ev["validation_1"][met], color="#D9A441", label="Prueba")
        ax.set_xlabel("Ronda de boosting (árbol n.°)"); ax.set_ylabel(met); ax.set_title("Cada árbol corrige el error")
        ax.legend(fontsize=8)
        st.pyplot(fig)
    if k == "D4":
        st.write("Matriz de confusión [[VN, FP], [FN, VP]]:", confusion_matrix(yte, pred).tolist())


# ---- D1
with tabs[1]:
    st.subheader(DIM["D1"]["nombre"] + " — predecir Yield (t)")
    bloque_modelo("D1")
    st.divider()
    st.subheader("Escenarios: ¿qué pasa con la producción total si cambia el riego?")
    m = MODELOS["D1"][0]; F = DIM["D1"]["feats"]
    col_a, col_b = st.columns(2)
    pes = col_a.selectbox("Riego en escenario pesimista", sorted(df.Irrigation_Type.unique()), index=sorted(df.Irrigation_Type.unique()).index("Rain-fed"), key="p1")
    opt = col_b.selectbox("Riego en escenario optimista", sorted(df.Irrigation_Type.unique()), index=sorted(df.Irrigation_Type.unique()).index("Drip"), key="o1")
    base = m.predict(df[F]).sum()
    esc = {"Pesimista": m.predict(df[F].assign(Irrigation_Type=pes)).sum(), "Base": base,
           "Optimista": m.predict(df[F].assign(Irrigation_Type=opt)).sum()}
    t = pd.DataFrame({"Producción total (t)": esc}); t["Variación (%)"] = (t.iloc[:, 0] / base - 1) * 100
    a, b = st.columns([1, 1.2])
    a.dataframe(t.style.format({"Producción total (t)": "{:,.0f}", "Variación (%)": "{:+.1f}"}), use_container_width=True)
    b.pyplot(grafico_barras_esc(t, "Producción total (t)", "Producción total por escenario (t)"))

# ---- D2
with tabs[2]:
    st.subheader(DIM["D2"]["nombre"] + " — predecir Water_Usage (m³)")
    bloque_modelo("D2")
    st.divider()
    st.subheader("Escenarios: productividad hídrica (t por cada 1,000 m³)")
    st.caption("Productividad hídrica = Yield predicho (modelo D1) ÷ agua predicha (modelo D2) × 1,000")
    m1, m2 = MODELOS["D1"][0], MODELOS["D2"][0]
    F1, F2 = DIM["D1"]["feats"], DIM["D2"]["feats"]
    filas = {}
    for nombre, riego in [("Pesimista", "Rain-fed"), ("Base", None), ("Optimista", "Drip")]:
        Z1, Z2 = (df[F1], df[F2]) if riego is None else (df[F1].assign(Irrigation_Type=riego), df[F2].assign(Irrigation_Type=riego))
        agua, prod = m2.predict(Z2).sum(), m1.predict(Z1).sum()
        filas[nombre] = {"Agua total (m³)": agua, "Producción (t)": prod, "Productividad hídrica": prod / agua * 1000}
    t = pd.DataFrame(filas).T
    t["Variación productividad (%)"] = (t["Productividad hídrica"] / t.loc["Base", "Productividad hídrica"] - 1) * 100
    a, b = st.columns([1.3, 1])
    a.dataframe(t.style.format({"Agua total (m³)": "{:,.0f}", "Producción (t)": "{:,.0f}", "Productividad hídrica": "{:.2f}",
                                "Variación productividad (%)": "{:+.1f}"}), use_container_width=True)
    b.pyplot(grafico_barras_esc(t, "Productividad hídrica", "t por 1,000 m³", "{:.2f}"))
    st.info("El agua depende casi solo del área (importancia ≈ 60 %): el tipo de riego casi no cambia los m³ usados, "
            "pero sí cuánto se produce con ellos. La mejora está en la productividad del agua, no en el volumen.")

# ---- D3
with tabs[3]:
    st.subheader(DIM["D3"]["nombre"] + " — predecir Yield_per_acre (t/acre)")
    bloque_modelo("D3")
    st.divider()
    st.subheader("Escenarios: cambiar la dosis de fertilizante y pesticida")
    m = MODELOS["D3"][0]; F = DIM["D3"]["feats"]
    c1, c2 = st.columns(2)
    cambio_f = c1.slider("Cambio de fertilizante en el optimista (%)", 0, 50, 20, 5)
    cambio_p = c2.slider("Reducción de pesticida en el optimista (%)", 0, 50, 20, 5)
    def con_insumos(kf, kp):
        Z = df[F].copy()
        Z["Fertilizer"] = (Z["Fertilizer"] * kf).clip(upper=df["Fertilizer"].max())
        Z["Pesticide"] = Z["Pesticide"] * kp
        return m.predict(Z).mean()
    base = m.predict(df[F]).mean()
    esc = {"Pesimista": con_insumos(1 - cambio_f / 100, 1 + cambio_p / 100), "Base": base,
           "Optimista": con_insumos(1 + cambio_f / 100, 1 - cambio_p / 100)}
    t = pd.DataFrame({"t/acre promedio": esc}); t["Variación (%)"] = (t.iloc[:, 0] / base - 1) * 100
    a, b = st.columns([1, 1.2])
    a.dataframe(t.style.format({"t/acre promedio": "{:.4f}", "Variación (%)": "{:+.2f}"}), use_container_width=True)
    b.pyplot(grafico_barras_esc(t, "t/acre promedio", "Rendimiento por acre por escenario", "{:.3f}"))
    st.info("Los insumos mueven poco el rendimiento por acre (±2 %); el cultivo y el riego pesan mucho más "
            "(ver importancia). Conclusión: aumentar fertilizante no garantiza más rendimiento.")

# ---- D4
with tabs[4]:
    st.subheader(DIM["D4"]["nombre"] + " — probabilidad de productividad alta")
    bloque_modelo("D4")
    st.divider()
    st.subheader("Escenarios: elegir la temporada de siembra")
    m = MODELOS["D4"][0]; F = DIM["D4"]["feats"]
    temporadas = sorted(df.Season.unique())
    P = np.column_stack([m.predict_proba(df[F].assign(Season=s))[:, 1] for s in temporadas])
    pb = m.predict_proba(df[F])[:, 1]
    t = pd.DataFrame({"Prob. promedio": [P.min(1).mean(), pb.mean(), P.max(1).mean()],
                      "Fincas con prob. ≥ 50 %": [(P.min(1) >= .5).sum(), (pb >= .5).sum(), (P.max(1) >= .5).sum()]},
                     index=["Pesimista", "Base", "Optimista"])
    a, b = st.columns([1, 1.2])
    a.dataframe(t.style.format({"Prob. promedio": "{:.3f}"}), use_container_width=True)
    b.pyplot(grafico_barras_esc(t, "Fincas con prob. ≥ 50 %", "Fincas con productividad alta esperada"))
    st.subheader("Simulador")
    s1, s2, s3 = st.columns(3)
    cultivo = s1.selectbox("Cultivo", sorted(df.Crop_Type.unique()))
    suelo = s2.selectbox("Suelo", sorted(df.Soil_Type.unique()))
    temp = s3.selectbox("Temporada", temporadas)
    prob = m.predict_proba(pd.DataFrame([{"Crop_Type": cultivo, "Soil_Type": suelo, "Season": temp}]))[0, 1]
    st.metric("Probabilidad de productividad alta", f"{prob:.0%}")
    st.markdown("**Mapa de calor: probabilidad por cultivo y suelo (temporada elegida)**")
    grid = pd.DataFrame([[m.predict_proba(pd.DataFrame([{"Crop_Type": c, "Soil_Type": s, "Season": temp}]))[0, 1]
                          for s in sorted(df.Soil_Type.unique())] for c in sorted(df.Crop_Type.unique())],
                        index=sorted(df.Crop_Type.unique()), columns=sorted(df.Soil_Type.unique()))
    st.dataframe(grid.style.format("{:.0%}").background_gradient(cmap="Greens"), use_container_width=True)

# ---- Explicación
with tabs[5]:
    st.markdown("""
**XGBoost (Extreme Gradient Boosting)** construye árboles de decisión **en secuencia**:

1. Empieza con una predicción simple (el promedio).
2. Calcula el **error (residuo)** de cada finca.
3. Entrena un árbol pequeño (max_depth = 3) que aprende a **predecir ese error**.
4. Suma ese árbol a la predicción, multiplicado por el **learning_rate** (0.03): da pasos pequeños para no sobreajustar.
5. Repite n_estimators veces (500). Cada árbol nuevo **corrige lo que los anteriores no lograron**.

**Diferencia con Random Forest:** RF entrena árboles **independientes en paralelo** y los **promedia** (bagging);
XGBoost los entrena **uno después de otro**, cada uno corrigiendo al anterior (boosting). Además, XGBoost incluye
**regularización** (penaliza árboles muy complejos).

El gráfico “Cada árbol corrige el error” de cada pestaña muestra cómo baja el error ronda a ronda.
""")
