"""
Página 0 · Regresión lineal (lineal simple, Ridge y Lasso) en las dimensiones 1, 2 y 3.

Usa las mismas variables, la misma partición 80/20 (random_state = 42) y el mismo preprocesamiento
(estandarizar numéricas + one-hot de categóricas) que analisis_final.py, por eso las cifras coinciden
con el póster y el informe.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Regresión lineal – Caso agrícola", layout="wide", page_icon="📏")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from estilo import AYUDA, explica, guia, hero  # noqa: E402

R = json.load(open(os.path.join(RAIZ, "results.json"), encoding="utf-8"))
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#C0562A", "#1baf7a", "#a8a7a2"

# ---------------------------------------------------------------- datos (mismo tratamiento que el análisis)
df = pd.read_csv(os.path.join(RAIZ, "agriculture_dataset_synthetic.csv")).rename(columns={
    "Farm_Area(acres)": "Farm_Area", "Fertilizer_Used(tons)": "Fertilizer", "Pesticide_Used(kg)": "Pesticide",
    "Yield(tons)": "Yield", "Water_Usage(cubic meters)": "Water_Usage"})
df["Yield_per_acre"] = df["Yield"] / df["Farm_Area"]
CAT = ["Crop_Type", "Irrigation_Type", "Soil_Type", "Season"]
DIM = {
    "D1": dict(nombre="1 · Producción total", target="Yield", unidad="t",
               feats=["Farm_Area", "Crop_Type", "Irrigation_Type", "Fertilizer", "Pesticide", "Soil_Type", "Season", "Water_Usage"],
               num=["Farm_Area", "Fertilizer", "Pesticide", "Water_Usage"],
               pregunta="¿Cuántas toneladas producirá cada finca?"),
    "D2": dict(nombre="2 · Agua usada", target="Water_Usage", unidad="m³",
               feats=["Farm_Area", "Crop_Type", "Irrigation_Type", "Soil_Type", "Season"], num=["Farm_Area"],
               pregunta="¿Cuántos m³ de agua usará cada finca?"),
    "D3": dict(nombre="3 · Rendimiento por acre", target="Yield_per_acre", unidad="t/acre",
               feats=["Fertilizer", "Pesticide", "Crop_Type", "Irrigation_Type", "Soil_Type", "Season"], num=["Fertilizer", "Pesticide"],
               pregunta="¿Cuánto rinde cada acre según los insumos y el manejo?"),
}
ALPHAS = np.logspace(-3, 3, 60)


def prep(cfg):
    return ColumnTransformer([("num", StandardScaler(), cfg["num"]),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), [f for f in cfg["feats"] if f in CAT])])


def nombres(p):
    return [n.replace("num__", "").replace("cat__", "") for n in p.named_steps["prep"].get_feature_names_out()]


@st.cache_resource(show_spinner="Entrenando las regresiones…")
def entrenar(k):
    cfg = DIM[k]
    a, b, c, d = train_test_split(df[cfg["feats"]], df[cfg["target"]], test_size=0.2, random_state=42)
    mods = {"Regresión lineal": LinearRegression(), "Ridge (L2)": RidgeCV(alphas=ALPHAS),
            "Lasso (L1)": LassoCV(alphas=ALPHAS, max_iter=20000, random_state=42)}
    out = {}
    for n, m in mods.items():
        p = Pipeline([("prep", prep(cfg)), ("m", m)]).fit(a, c)
        out[n] = dict(pipe=p, pred=p.predict(b), r2=r2_score(d, p.predict(b)), r2_tr=r2_score(c, p.predict(a)),
                      rmse=mean_squared_error(d, p.predict(b)) ** .5, mae=mean_absolute_error(d, p.predict(b)))
    return out, (a, b, c, d)


# ---------------------------------------------------------------- encabezado
hero("📏 Regresión lineal, Ridge y Lasso",
     "El modelo más simple: traza una “línea” que relaciona cada variable con el resultado. Es la referencia contra la que "
     "comparamos Random Forest y XGBoost, y en agua e insumos resulta ser el mejor.", img="lineal")
guia(["**1️⃣ ¿Qué es?:** la idea con un ejemplo cotidiano y las fórmulas.",
      "**2️⃣ Resultados:** elija una dimensión y vea qué tanto acierta cada regresión (pase el cursor por el **?** de cada número).",
      "**3️⃣ ¿Qué variables pesan?:** los coeficientes muestran cuánto sube o baja el resultado con cada variable. "
      "Mueva el control de λ para ver cómo Lasso “apaga” variables.",
      "**4️⃣ Simulador:** arme una finca y vea cuánto produciría según la regresión.",
      "**5️⃣ Comparación:** las tres técnicas que pidió el curso (lineal, Random Forest y XGBoost) lado a lado."])

t1, t2, t3, t4, t5 = st.tabs(["1️⃣ ¿Qué es?", "2️⃣ Resultados", "3️⃣ ¿Qué variables pesan?", "4️⃣ Simulador",
                              "5️⃣ Lineal vs. Random Forest vs. XGBoost"])

# ---------------------------------------------------------------- 1. concepto
with t1:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("La idea")
        st.markdown(
            "Imagine que cada variable es un **botón** que suma o resta toneladas: “cada acre suma tanto”, “el riego por goteo "
            "suma tanto”, “el suelo arenoso resta tanto”. La regresión lineal busca el valor de cada botón (**coeficiente β**) "
            "para que la suma se parezca lo más posible a la producción real.")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p")
        st.markdown("**Ridge y Lasso** son la misma idea con un **freno (λ)** para que los botones no se exageren:")
        st.latex(r"\text{Ridge: } \min \sum (y_i-\hat{y}_i)^2 + \lambda \sum \beta_j^2")
        st.latex(r"\text{Lasso: } \min \sum (y_i-\hat{y}_i)^2 + \lambda \sum |\beta_j|")
        explica("Ridge achica todos los botones un poco. Lasso puede dejar algunos en **cero**: es como decir “esta variable "
                "no aporta, la apago”. Por eso Lasso también sirve para **seleccionar variables**.")
    with c2:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.scatter(df.Farm_Area, df.Yield, s=10, alpha=.5, color=AZUL, label="Fincas")
        m, b0 = np.polyfit(df.Farm_Area, df.Yield, 1)
        xs = np.linspace(df.Farm_Area.min(), df.Farm_Area.max(), 50)
        ax.plot(xs, m * xs + b0, color=NARANJA, lw=2.5, label=f"Línea: {m:.2f} t por acre")
        ax.set_xlabel("Área (acres)"); ax.set_ylabel("Producción (t)"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("Ejemplo: producción vs. área"); fig.tight_layout(); st.pyplot(fig)
        explica(f"Cada punto es una finca. La línea naranja dice que, en promedio, **cada acre adicional suma {m:.2f} t**. "
                "La regresión completa hace lo mismo con todas las variables a la vez.")

# ---------------------------------------------------------------- 2. resultados
with t2:
    k = st.radio("Dimensión", list(DIM), format_func=lambda x: DIM[x]["nombre"], horizontal=True,
                 help="Cada dimensión es una pregunta distinta del caso. D1 producción, D2 agua, D3 rendimiento por acre.")
    cfg = DIM[k]; RES, (a, b, c, d) = entrenar(k)
    st.markdown(f"**Pregunta:** {cfg['pregunta']}")
    cols = st.columns(3)
    for col, (n, r) in zip(cols, RES.items()):
        with col:
            st.markdown(f"**{n}**")
            st.metric("R² (prueba)", f"{r['r2']:.3f}", f"entrenamiento {r['r2_tr']:.3f}", delta_color="off", help=AYUDA["R2"] + " " + AYUDA["train"])
            st.metric("Error promedio (MAE)", f"{r['mae']:,.4g} {cfg['unidad']}", help=AYUDA["MAE"])
            st.metric("RMSE", f"{r['rmse']:,.4g} {cfg['unidad']}", help=AYUDA["RMSE"])
    mejor = max(RES, key=lambda n: RES[n]["r2"])
    explica(f"Las tres regresiones aciertan casi igual (R² ≈ {RES[mejor]['r2']:.2f}): explican cerca del "
            f"{RES[mejor]['r2'] * 100:.0f} % de las diferencias entre fincas. El R² de entrenamiento es parecido al de prueba, así "
            "que **no memorizan**: aprenden una regla general.")
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(4.8, 4))
        ax.scatter(d, RES[mejor]["pred"], s=14, alpha=.6, color=AZUL)
        lim = [min(d.min(), RES[mejor]["pred"].min()), max(d.max(), RES[mejor]["pred"].max())]
        ax.plot(lim, lim, "--", color=NARANJA, lw=1.5)
        ax.set_xlabel(f"Real ({cfg['unidad']})"); ax.set_ylabel(f"Predicho ({cfg['unidad']})"); ax.set_title(f"{mejor}: predicho vs. real")
        fig.tight_layout(); st.pyplot(fig)
        explica("Cada punto es una finca del 20 % de prueba (el modelo no la vio al aprender). Si el punto cae sobre la "
                "línea punteada, la predicción fue exacta; mientras más pegados a la línea, mejor.")
    with g2:
        err = d - RES[mejor]["pred"]
        fig, ax = plt.subplots(figsize=(4.8, 4))
        ax.hist(err, bins=25, color=VERDE, alpha=.85); ax.axvline(0, color=NARANJA, lw=1.5)
        ax.set_xlabel(f"Error = real − predicho ({cfg['unidad']})"); ax.set_ylabel("N.° de fincas"); ax.set_title("¿Cuánto se equivoca?")
        fig.tight_layout(); st.pyplot(fig)
        explica("La mayoría de errores está cerca de cero y repartidos a ambos lados: el modelo no se equivoca siempre "
                "hacia arriba ni siempre hacia abajo.")

# ---------------------------------------------------------------- 3. coeficientes y Lasso
with t3:
    k3 = st.radio("Dimensión ", list(DIM), format_func=lambda x: DIM[x]["nombre"], horizontal=True, index=2,
                  help="Elija qué pregunta quiere analizar.")
    cfg = DIM[k3]; RES, (a, b, c, d) = entrenar(k3)
    lam_cv = RES["Lasso (L1)"]["pipe"].named_steps["m"].alpha_
    opciones = sorted(set([float(f"{x:.3g}") for x in np.logspace(np.log10(lam_cv) - 2, np.log10(lam_cv) + 2, 17)] + [float(f"{lam_cv:.3g}")]))
    lam = st.select_slider("Fuerza del freno de Lasso (λ)", options=opciones, value=float(f"{lam_cv:.3g}"), help=AYUDA["alpha"])
    las = Pipeline([("prep", prep(cfg)), ("m", Lasso(alpha=lam, max_iter=20000))]).fit(a, c)
    coef = pd.Series(las.named_steps["m"].coef_, index=nombres(las))
    lin = pd.Series(RES["Regresión lineal"]["pipe"].named_steps["m"].coef_, index=nombres(RES["Regresión lineal"]["pipe"]))
    c1, c2 = st.columns([1.4, 1])
    with c1:
        top = lin.abs().sort_values(ascending=False).head(14).index
        fig, ax = plt.subplots(figsize=(6.4, 5))
        yy = np.arange(len(top))
        ax.barh(yy + .2, lin[top].values, height=.4, color=GRIS, label="Regresión lineal (sin freno)")
        ax.barh(yy - .2, coef[top].values, height=.4, color=[NARANJA if v != 0 else "#e5e5e5" for v in coef[top].values],
                label=f"Lasso (λ = {lam:g})")
        ax.set_yticks(yy); ax.set_yticklabels(top, fontsize=8); ax.invert_yaxis(); ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel(f"Coeficiente (cambio en {cfg['unidad']})"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("¿Cuánto suma o resta cada variable?"); fig.tight_layout(); st.pyplot(fig)
    with c2:
        st.metric("Variables apagadas por Lasso (coef. = 0)", f"{int((coef == 0).sum())} de {len(coef)}",
                  help="Con más freno (λ más alto), Lasso deja en cero las variables que menos aportan.")
        st.metric("R² de Lasso con este λ", f"{r2_score(d, las.predict(b)):.3f}", help=AYUDA["R2"])
        st.caption(f"λ elegido por validación cruzada: {lam_cv:.3g}")
        explica("Barra hacia la derecha = la variable **sube** el resultado; hacia la izquierda = lo **baja**. "
                "Las numéricas están estandarizadas: el coeficiente es el cambio cuando la variable sube “una desviación típica”. "
                "Las categóricas (cultivo, riego…) se comparan contra el promedio de su grupo.")
        explica("Suba λ y verá que las barras naranjas se achican y algunas se vuelven grises (cero): así Lasso **selecciona** "
                "solo las variables que importan, sin perder mucha precisión.")

# ---------------------------------------------------------------- 4. simulador (D3 × área)
with t4:
    st.subheader("Arme una finca y vea su producción esperada")
    RES3, _ = entrenar("D3")
    p3 = RES3["Regresión lineal"]["pipe"]
    s1, s2, s3 = st.columns(3)
    cultivo = s1.selectbox("Cultivo", sorted(df.Crop_Type.unique()), help=AYUDA["cultivo"])
    suelo = s1.selectbox("Suelo", sorted(df.Soil_Type.unique()), help=AYUDA["suelo"])
    riego = s2.selectbox("Tipo de riego", sorted(df.Irrigation_Type.unique()), help=AYUDA["riego"])
    temp = s2.selectbox("Temporada", sorted(df.Season.unique()), help=AYUDA["temporada"])
    area = s3.number_input("Área (acres)", 5.0, 500.0, float(round(df.Farm_Area.median())), 5.0, help=AYUDA["area"])
    fert = s3.slider("Fertilizante (t)", 0.5, 10.0, float(round(df.Fertilizer.mean(), 1)), 0.1, help=AYUDA["fert"])
    pest = s3.slider("Pesticida (kg)", 0.1, 5.0, float(round(df.Pesticide.mean(), 1)), 0.1, help=AYUDA["pest"])
    fila = pd.DataFrame([dict(Fertilizer=fert, Pesticide=pest, Crop_Type=cultivo, Irrigation_Type=riego, Soil_Type=suelo, Season=temp)])
    tpa = float(p3.predict(fila)[0])
    m1, m2 = st.columns(2)
    m1.metric("Rendimiento por acre esperado", f"{tpa:.3f} t/acre", f"{(tpa / df.Yield_per_acre.mean() - 1) * 100:+.1f} % vs. promedio",
              help="Lo que predice la regresión lineal de la D3 para esta combinación.")
    m2.metric("Producción total esperada", f"{tpa * area:,.1f} t", help="Rendimiento por acre × área (la misma idea del modelo “por acre × área”).")
    filas = []
    for rr in sorted(df.Irrigation_Type.unique()):
        v = float(p3.predict(fila.assign(Irrigation_Type=rr))[0])
        filas.append({"Riego": rr, "t/acre": round(v, 3), "Producción (t)": round(v * area, 1)})
    st.dataframe(pd.DataFrame(filas).sort_values("t/acre", ascending=False), hide_index=True, use_container_width=True)
    explica("La tabla muestra **la misma finca** cambiando solo el riego. Así se ve el efecto de esa decisión con todo lo demás igual: "
            "goteo (Drip) suele quedar arriba y secano (Rain-fed) abajo.")

# ---------------------------------------------------------------- 5. comparación con RF y XGBoost
with t5:
    st.subheader("Las tres técnicas del curso en las mismas condiciones (20 % de prueba)")
    filas = []
    for k_, cfg in DIM.items():
        m = R["dims"][k_]["modelos"]
        fila = {"Dimensión": cfg["nombre"], "Regresión lineal": m["Regresión lineal"]["R2"],
                "Random Forest": m["Random Forest"]["R2"], "XGBoost": m["XGBoost"]["R2"]}
        fila["Gana"] = max(["Regresión lineal", "Random Forest", "XGBoost"], key=lambda n: fila[n])
        filas.append(fila)
    mj = R["d1_mejorado"]
    T = pd.DataFrame(filas)
    st.dataframe(T.style.format({c: "{:.3f}" for c in ["Regresión lineal", "Random Forest", "XGBoost"]})
                 .highlight_max(axis=1, subset=["Regresión lineal", "Random Forest", "XGBoost"], color="#FFE3C2"),
                 hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(3); w = .26
    for i, (n, col) in enumerate([("Regresión lineal", VERDE), ("Random Forest", AZUL), ("XGBoost", NARANJA)]):
        vals = T[n].values
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=n)
        for xi, v in zip(x + (i - 1) * w, vals): ax.text(xi, v + .005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(T["Dimensión"]); ax.set_ylim(0.75, 1.0); ax.set_ylabel("R² en prueba")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right"); fig.tight_layout(); st.pyplot(fig)
    st.caption(f"Con el truco “por acre × área” en la producción total: Random Forest R² = {mj['Random Forest (por acre × área)']['R2']:.3f} · "
               f"XGBoost R² = {mj['XGBoost (por acre × área)']['R2']:.3f}.")
    st.caption("El eje vertical empieza en 0.75 para que se vean las diferencias; todas las barras superan con amplitud al modelo base (R² ≈ 0).")
    explica("En **agua (D2)** e **insumos (D3)** la regresión lineal gana: la relación es casi una línea recta y no hace falta "
            "un modelo más complejo. En **producción (D1)** gana XGBoost porque capta combinaciones (cultivo × riego × tamaño); "
            "con el truco “por acre × área” llega a R² 0.975. Conclusión: se elige el modelo más simple que cumple la meta.")
