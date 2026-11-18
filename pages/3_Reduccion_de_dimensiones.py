"""
PROTOTIPO – REDUCCIÓN DE DIMENSIONES DEL CASO AGRÍCOLA (Random Forest)
Curso: Modelamiento predictivo de datos – Grupo 3

Ejecutar:
    pip install streamlit pandas numpy scikit-learn matplotlib
    streamlit run app_reduccion_caso.py
Necesita agriculture_dataset_synthetic.csv en la misma carpeta (o subirlo).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Reducción de dimensiones – Caso agrícola", layout="wide")

# ---------------------------------------------------------------- ENCABEZADO CON NOMBRE (para la captura)
nombre = st.sidebar.text_input("Nombre para el encabezado", "Grupo 3 · Bajalqui De la Cruz, L. M. · De La Cuba Oliva, F. A. · Perdicci Taboada, G. G.")
st.markdown(f"<div style='background:#2C5F2D;color:white;padding:10px 16px;border-radius:8px'>"
            f"<b>{nombre}</b> · Modelamiento predictivo de datos · Reducción de dimensiones del caso agrícola</div>",
            unsafe_allow_html=True)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # para importar estilo.py
from estilo import hero, guia  # noqa: E402
hero("📉 Reducción de dimensiones: ¿cuántas variables necesito?",
     "Área y agua dicen casi lo mismo. Aquí vemos qué variables sobran (VIF), cómo resumirlas (PCA) y si el modelo "
     "pierde precisión al usar menos.", img="trigo")
guia(["**1️⃣ Diagnóstico:** el mapa de correlación y el VIF muestran qué variables se repiten (VIF > 10 = redundante).",
      "**2️⃣ PCA:** cuánta información guarda cada componente; con 3 componentes se conserva casi el 99 %.",
      "**3️⃣ Selección:** qué variables pesan más en el modelo y cuáles se pueden dejar de lado.",
      "**4️⃣ ¿Se pierde precisión?:** compare el R² con todas las variables, sin la redundante y con PCA."])

# ---------------------------------------------------------------- 1. DATOS
archivo = st.sidebar.file_uploader("CSV (opcional)", type=["csv"])
RUTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agriculture_dataset_synthetic.csv")  # el CSV está en la carpeta raíz de la app
df = pd.read_csv(archivo) if archivo is not None else (pd.read_csv(RUTA) if os.path.exists(RUTA) else None)
if df is None:
    st.warning("Coloca agriculture_dataset_synthetic.csv junto a la app o súbelo."); st.stop()
df = df.rename(columns={"Farm_Area(acres)": "Farm_Area", "Fertilizer_Used(tons)": "Fertilizer",
                        "Pesticide_Used(kg)": "Pesticide", "Yield(tons)": "Yield",
                        "Water_Usage(cubic meters)": "Water_Usage"})
NUM = ["Farm_Area", "Fertilizer", "Pesticide", "Water_Usage"]           # variables numéricas predictoras
CAT = ["Crop_Type", "Irrigation_Type", "Soil_Type", "Season"]
y = df["Yield"]                                                         # variable objetivo

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Diagnóstico (correlación y VIF)", "2️⃣ PCA", "3️⃣ Selección de variables", "4️⃣ ¿Se pierde precisión?"])

# ---------------------------------------------------------------- 2. DIAGNÓSTICO: ¿hay información repetida?
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        corr = df[NUM + ["Yield"]].corr()                               # correlación de Pearson
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
        for (i, j), v in np.ndenumerate(corr.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im); ax.set_title("Matriz de correlación"); fig.tight_layout(); st.pyplot(fig)
    with c2:
        vif = []                                                        # VIF = 1/(1-R²) de cada variable contra las demás
        for col in NUM:
            otras = [c for c in NUM if c != col]
            r2 = LinearRegression().fit(df[otras], df[col]).score(df[otras], df[col])
            vif.append({"Variable": col, "VIF": 1 / (1 - r2)})
        vif = pd.DataFrame(vif).set_index("Variable")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(vif.index, vif["VIF"], color=["#B23A48" if v > 10 else "#97BC62" for v in vif["VIF"]])
        ax.axvline(10, color="red", ls="--"); ax.set_title("VIF (rojo = mayor que 10)"); st.pyplot(fig)
        st.dataframe(vif.style.format("{:.2f}"))
    st.info("Farm_Area y Water_Usage se correlacionan 0.95 y tienen VIF ≈ 11: dicen casi lo mismo. "
            "Eso justifica reducir dimensiones.")

# ---------------------------------------------------------------- 3. PCA
with tab2:
    X_std = StandardScaler().fit_transform(df[NUM])                     # PASO 1: estandarizar
    pca = PCA().fit(X_std)                                              # PASOS 2–4: covarianza, autovectores, ordenar
    var = pca.explained_variance_ratio_
    acum = np.cumsum(var)
    k = st.slider("Componentes a conservar", 1, len(NUM), 3)
    st.metric("Varianza retenida", f"{acum[k-1]*100:.2f}%")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(range(1, len(NUM) + 1), var, color="#97BC62", label="Individual")
        ax.plot(range(1, len(NUM) + 1), acum, "o-", color="#2C5F2D", label="Acumulada")
        ax.axhline(0.95, color="red", ls="--", lw=1, label="95 %")
        ax.set_xlabel("Componente"); ax.set_ylabel("Varianza explicada"); ax.legend(); ax.set_title("Varianza explicada por PCA")
        st.pyplot(fig)
    with c2:
        cargas = pd.DataFrame(pca.components_.T, index=NUM, columns=[f"PC{i+1}" for i in range(len(NUM))])
        st.write("**Cargas:** qué variables forman cada componente")
        st.dataframe(cargas.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None))
    Z = pca.transform(X_std)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", alpha=0.6, s=15)
    for i, v in enumerate(NUM):                                         # biplot: flechas = variables originales
        ax.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3, color="red", head_width=0.08)
        ax.text(pca.components_[0, i] * 3.3, pca.components_[1, i] * 3.3, v, color="red")
    fig.colorbar(sc, label="Yield (t)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Biplot PC1 vs PC2 (color = rendimiento)")
    st.pyplot(fig)
    st.info("PC1 junta área y agua (el 'tamaño' de la finca) y se asocia al rendimiento; "
            "con 3 componentes se conserva casi toda la información.")

# ---------------------------------------------------------------- 4. SELECCIÓN DE VARIABLES
with tab3:
    X_all = pd.get_dummies(df[NUM + CAT], columns=CAT).astype(float)
    sel = SelectKBest(f_regression, k="all").fit(X_all, y)              # F de regresión de cada variable con Yield
    puntaje = pd.Series(sel.scores_, index=X_all.columns).sort_values().tail(12)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_all, y)
    imp = pd.Series(rf.feature_importances_, index=X_all.columns).sort_values().tail(12)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4.5)); ax.barh(puntaje.index, puntaje.values, color="#97BC62")
        ax.set_title("SelectKBest (F de regresión) – top 12"); fig.tight_layout(); st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4.5)); ax.barh(imp.index, imp.values, color="#2C5F2D")
        ax.set_title("Importancia en Random Forest – top 12"); fig.tight_layout(); st.pyplot(fig)

# ---------------------------------------------------------------- 5. ¿SE PIERDE PRECISIÓN AL REDUCIR?
with tab4:
    X_all = pd.get_dummies(df[NUM + CAT], columns=CAT).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.2, random_state=42)
    sc_ = StandardScaler().fit(Xtr[NUM]); p_ = PCA(n_components=k).fit(sc_.transform(Xtr[NUM]))

    def con_pca(D):
        comp = pd.DataFrame(p_.transform(sc_.transform(D[NUM])), index=D.index, columns=[f"PC{i+1}" for i in range(k)])
        return pd.concat([comp, D.drop(columns=NUM)], axis=1)

    escenarios = {
        "Todas las variables": (Xtr, Xte),
        "Sin Water_Usage (VIF > 10)": (Xtr.drop(columns="Water_Usage"), Xte.drop(columns="Water_Usage")),
        f"PCA ({k} componentes) + categóricas": (con_pca(Xtr), con_pca(Xte)),
    }
    filas = []
    for n, (a, b) in escenarios.items():
        m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(a, ytr)
        p = m.predict(b)
        filas.append({"Escenario": n, "N.° variables": a.shape[1], "R²": r2_score(yte, p),
                      "RMSE (t)": mean_squared_error(yte, p) ** 0.5, "MAE (t)": mean_absolute_error(yte, p)})
    t = pd.DataFrame(filas).set_index("Escenario")
    st.dataframe(t.style.format({"R²": "{:.3f}", "RMSE (t)": "{:.2f}", "MAE (t)": "{:.2f}"}))
    fig, ax = plt.subplots(figsize=(7, 3)); ax.barh(t.index, t["R²"], color="#2C5F2D"); ax.set_xlim(0.8, 1)
    ax.set_title("R² de Random Forest según la reducción"); st.pyplot(fig)
    st.success("Conclusión: se puede trabajar con menos variables sin perder precisión, porque la información "
               "repetida (área ≈ agua) no aporta nada nuevo al modelo.")
