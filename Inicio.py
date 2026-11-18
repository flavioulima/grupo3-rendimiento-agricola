"""
APP GRUPAL – Modelamiento Predictivo de Datos · Grupo 3
Caso: Análisis y predicción del rendimiento agrícola (600 explotaciones)

Página de inicio. Las demás páginas están en la carpeta pages/:
  1_Random_Forest.py            · prototipo con 3 escenarios del modelo + pesimista/base/optimista
  2_XGBoost.py                  · 4 dimensiones con XGBoost y escenarios
  3_Reduccion_de_dimensiones.py · correlación, VIF, PCA y precisión con menos variables

Ejecutar en local:   streamlit run Inicio.py
Publicar:            Streamlit Community Cloud (ver README.md)
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Grupo 3 · Rendimiento agrícola", page_icon="🌾", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(BASE, "results.json"), encoding="utf-8"))   # cifras calculadas por analisis_final.py

st.markdown("<div style='background:#2C5F2D;color:white;padding:10px 16px;border-radius:8px'>"
            "<b>Grupo 3</b> · Bajalqui De la Cruz, Luz María · De La Cuba Oliva, Flavio Alessandro · "
            "Perdicci Taboada, Gianfranco Giovanni<br>Modelamiento Predictivo de Datos · Universidad de Lima · "
            "Docente: Taquía Gutiérrez, José Antonio</div>", unsafe_allow_html=True)
st.title("🌾 Predicción del rendimiento agrícola con Machine Learning")
st.write("Sistema de apoyo a decisiones para planificar la cosecha, el riego, los insumos y la elección de "
         "cultivo y temporada en 600 explotaciones agrícolas. Use el menú de la izquierda para abrir cada prototipo.")

# --------------------------------------------------------------- KPI del caso
k = R["kpi"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Producción total", f"{k['prod_total']:,.0f} t")
c2.metric("Rendimiento promedio", f"{k['tpa_medio']:.3f} t/acre")
c3.metric("Agua total (millones de m³)", f"{k['agua_total']/1e6:,.1f}")
c4.metric("Productividad hídrica (t/1,000 m³)", f"{k['prod_hidrica']:.2f}")

# --------------------------------------------------------------- Mejores modelos por dimensión
st.subheader("Resultados por dimensión (conjunto de prueba, 20 %)")
filas = []
for d, nom, obj in [("D1", "Productividad agrícola", "Yield (t)"), ("D2", "Gestión hídrica", "Water_Usage (m³)"),
                    ("D3", "Insumos agroquímicos", "Yield_per_acre (t/acre)")]:
    m = R["dims"][d]["modelos"]
    filas.append({"Dimensión": f"{d} · {nom}", "Objetivo": obj, "Métrica": "R²",
                  "Modelo base": round(m["Modelo base (media)"]["R2"], 3), "Lineal": round(m["Regresión lineal"]["R2"], 3),
                  "Random Forest": round(m["Random Forest"]["R2"], 3), "XGBoost": round(m["XGBoost"]["R2"], 3)})
m4 = R["dims"]["D4"]["modelos"]
filas.append({"Dimensión": "D4 · Cultivo y suelo", "Objetivo": "Productividad alta (1/0)", "Métrica": "AUC",
              "Modelo base": round(m4["Modelo base (clase mayoritaria)"]["AUC"], 3), "Lineal": round(m4["Regresión logística"]["AUC"], 3),
              "Random Forest": round(m4["Random Forest"]["AUC"], 3), "XGBoost": round(m4["XGBoost"]["AUC"], 3)})
st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
mj = R["d1_mejorado"]
st.success(f"Modelo mejorado de la D1 (predecir t/acre con variables de manejo × área): "
           f"Random Forest R² = {mj['Random Forest (por acre × área)']['R2']:.3f} · "
           f"XGBoost R² = {mj['XGBoost (por acre × área)']['R2']:.3f}.")

# --------------------------------------------------------------- Escenarios
st.subheader("Escenarios pesimista / base / optimista (variación frente a la situación actual)")
txt = {"D1": "Riego: todo Rain-fed / actual / todo goteo (producción total)",
       "D2": "Riego: productividad hídrica (t por 1,000 m³)",
       "D3": "Insumos: fertilizante −20 % y pesticida +20 % / actual / al revés (t/acre)",
       "D4": "Temporada de siembra: peor / actual / mejor (fincas con productividad alta)"}
filas = []
for d in ["D1", "D2", "D3", "D4"]:
    for n in ["Random Forest", "XGBoost"]:
        e = R["escenarios"][n][d]
        filas.append({"Dimensión": d, "Supuesto": txt[d], "Modelo": n,
                      "Pesimista": f"{(e['Pesimista']/e['Base']-1)*100:+.1f} %", "Optimista": f"{(e['Optimista']/e['Base']-1)*100:+.1f} %"})
st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

# --------------------------------------------------------------- Enfoque prescriptivo
st.subheader("Enfoque prescriptivo: escenarios de solución (XGBoost por acre × área)")
st.caption(f"S1: las {R['n_convertidas_riego']} fincas de secano o riego manual pasan a goteo · S2: +20 % de fertilizante · "
           "S3: cada finca siembra en la temporada de mayor rendimiento esperado · S4: S1 + S3. El agua se mantiene constante.")
PX = pd.DataFrame(R["prescriptivo"]["XGBoost (por acre × área)"]).T
kpi = st.selectbox("Indicador (KPI)", list(PX.columns))
var = (PX[kpi] / PX[kpi].iloc[0] - 1) * 100
tab = pd.DataFrame({"Escenario": PX.index, kpi: PX[kpi].round(3), "Variación vs. actual (%)": var.round(1)})
c1, c2 = st.columns([1.1, 1])
c1.dataframe(tab, hide_index=True, use_container_width=True)
c2.bar_chart(var.iloc[1:].rename("Variación vs. actual (%)"), horizontal=True)
mejor = var.iloc[1:].idxmax()
st.success(f"Escenario recomendado para «{kpi}»: **{mejor}** ({var[mejor]:+.1f} % frente a la situación actual).")

st.info("**Páginas:** 1 · Random Forest (3 escenarios del modelo y simulador) · 2 · XGBoost (4 dimensiones) · "
        "3 · Reducción de dimensiones (correlación, VIF, PCA). Datos: dataset sintético de 600 explotaciones; "
        "los escenarios son contrafactuales del modelo, no ensayos de campo.")
