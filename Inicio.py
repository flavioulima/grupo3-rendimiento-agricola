"""
APP GRUPAL – Modelamiento Predictivo de Datos · Grupo 3
Caso: Análisis, predicción y prescripción del rendimiento agrícola (600 explotaciones)

Página de inicio. Las demás páginas están en la carpeta pages/:
  0_Regresion_lineal.py         · regresión lineal, Ridge y Lasso (D1–D3), coeficientes, simulador y comparación
  1_Random_Forest.py            · prototipo con 3 escenarios del modelo + simulador "¿qué pasa si?"
  2_XGBoost.py                  · 4 dimensiones con XGBoost y escenarios
  3_Reduccion_de_dimensiones.py · correlación, VIF, PCA y precisión con menos variables
estilo.py                       · banner con imagen, tarjetas y guía de uso (compartido por todas las páginas)

Ejecutar en local:   streamlit run Inicio.py
Publicar:            Streamlit Community Cloud (ver README.md)
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Grupo 3 · Rendimiento agrícola", page_icon="🌾", layout="wide")
from estilo import IMG, guia, hero, kpi, tarjeta  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(BASE, "results.json"), encoding="utf-8"))   # cifras calculadas por analisis_final.py
D = R["dims"]
MJ = R["d1_mejorado"]["XGBoost (por acre × área)"]
PX = pd.DataFrame(R["prescriptivo"]["XGBoost (por acre × área)"]).T
ACT, S4 = PX.iloc[0], PX.iloc[-1]
K = list(PX.columns)
vp = lambda s, k: (PX.loc[s, k] / ACT[k] - 1) * 100


def barras(serie, eje, resaltar=None, fmt=".3f", orden=None):
    """Barras horizontales ordenadas de mayor a menor, con etiquetas completas y el valor al final de cada barra."""
    d = serie.reset_index(); d.columns = ["Nombre", "Valor"]
    d["Grupo"] = ["Destacado" if (resaltar and resaltar in n) else "Otros" for n in d["Nombre"]]
    base = alt.Chart(d).encode(
        y=alt.Y("Nombre:N", sort=orden or list(d.sort_values("Valor", ascending=False)["Nombre"]), title=None, axis=alt.Axis(labelLimit=320, labelFontSize=13)),
        x=alt.X("Valor:Q", title=eje),
        tooltip=["Nombre", alt.Tooltip("Valor:Q", format=fmt)])
    bar = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color("Grupo:N", scale=alt.Scale(domain=["Destacado", "Otros"], range=["#C0562A", "#2a78d6"]), legend=None))
    txt = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("Valor:Q", format=fmt))
    st.altair_chart((bar + txt).properties(height=34 * len(d) + 30), use_container_width=True)

# ------------------------------------------------------------------ Encabezado
hero("🌾 Predicción y prescripción del rendimiento agrícola",
     "Usamos machine learning con 600 fincas para predecir cuánto producen y para recomendar qué decisión "
     "de riego, insumos y temporada mejora más la producción.",
     pill=f"Plan recomendado: +{vp(PX.index[-1], K[0]):.1f} % de producción con la misma agua")
st.caption("Grupo 3 · Bajalqui De la Cruz, Luz María · De La Cuba Oliva, Flavio Alessandro · Perdicci Taboada, "
           "Gianfranco Giovanni · Modelamiento Predictivo de Datos · Universidad de Lima · Docente: Taquía Gutiérrez, José Antonio")

guia(["Empiece por **🏠 El caso**: qué problema resolvemos y con qué datos.",
      "En **📊 Resultados** vea, dimensión por dimensión, qué modelo acierta más y cómo se lee cada métrica.",
      "En **🧭 Escenarios** compare las decisiones S1–S4 y elija el indicador que quiera mejorar.",
      "En **📖 Glosario** están los términos técnicos explicados en una línea.",
      "En el menú de la izquierda están los cuatro prototipos: **Regresión lineal**, **Random Forest** (simulador), "
      "**XGBoost** (4 dimensiones) y **Reducción de dimensiones**.",
      "¿Ve un ícono **?** junto a un número o control? Pase el cursor encima: aparece un globo que explica qué es."])

t_caso, t_res, t_esc, t_glo = st.tabs(["🏠 El caso", "📊 Resultados", "🧭 Escenarios y recomendación", "📖 Glosario"])

# ------------------------------------------------------------------ 1. El caso
with t_caso:
    c_txt, c_img = st.columns([1.4, 1])
    with c_txt:
        st.subheader("¿Qué problema resolvemos?")
        st.markdown(
            "La producción de una finca depende de muchas cosas a la vez: **cultivo, suelo, tipo de riego, fertilizante, "
            "pesticida, agua y temporada**. El agricultor necesita saber dos cosas: **cuánto va a producir** y **qué "
            "decisión le conviene cambiar**. Los estudios recientes muestran que el *machine learning* captura estas "
            "relaciones mejor que los métodos tradicionales (van Klompenburg et al., 2020; Jeong et al., 2016).")
        st.markdown("**Objetivo:** predecir el rendimiento y **prescribir** la combinación de riego, insumos y temporada "
                    "que más mejora la producción y el uso del agua.")
    with c_img:
        st.image(IMG["suelo"], caption="El rendimiento depende del suelo, el agua, los insumos y el manejo.",
                 use_container_width=True)

    st.subheader("Situación actual de las 600 fincas")
    k = R["kpi"]
    c = st.columns(4)
    kpi(c[0], "Producción total", f"{k['prod_total']:,.0f} t")
    kpi(c[1], "Rendimiento promedio", f"{k['tpa_medio']:.3f} t/acre")
    kpi(c[2], "Agua usada", f"{k['agua_total']/1e6:,.1f} M m³")
    kpi(c[3], "Productividad hídrica", f"{k['prod_hidrica']:.2f} t/1,000 m³")

    st.subheader("Las 4 preguntas (dimensiones) que responde el proyecto")
    c1, c2 = st.columns(2)
    with c1:
        tarjeta("D1 · Productividad", f"¿Cuántas toneladas producirá cada finca? Meta R² ≥ 0.90 → logrado "
                f"<b>{MJ['R2']:.3f}</b> con XGBoost.")
        tarjeta("D3 · Insumos", f"¿Cuánto rinde cada acre según fertilizante, pesticida y manejo? Meta R² ≥ 0.85 → "
                f"<b>{D['D3']['modelos']['Regresión lineal']['R2']:.3f}</b> con regresión lineal.")
    with c2:
        tarjeta("D2 · Agua", f"¿Cuánta agua usará cada finca? Meta R² ≥ 0.85 → <b>{D['D2']['modelos']['Lasso (L1)']['R2']:.3f}</b> "
                "con Lasso.")
        tarjeta("D4 · Cultivo y suelo", f"¿La finca será de productividad alta (≥ {R['umbral']:.3f} t/acre)? Meta AUC ≥ 0.80 → "
                f"<b>{D['D4']['modelos']['XGBoost']['AUC']:.3f}</b> con XGBoost.")

    c_a, c_b = st.columns(2)
    c_a.image(IMG["trigo"], caption="Cada cultivo responde distinto a los mismos insumos.", use_container_width=True)
    c_b.image(IMG["riego"], caption="El tipo de riego cambia el rendimiento y el uso del agua.", use_container_width=True)
    with st.expander("📋 ¿Qué datos usamos?"):
        st.markdown("600 fincas (dataset sintético) con 10 variables: cultivo, área (acres), tipo de riego, fertilizante (t), "
                    "pesticida (kg), rendimiento (t), tipo de suelo, temporada (Kharif, Rabi, Zaid) y agua (m³). "
                    "No hay valores vacíos. Se entrena con el 80 % de las fincas y se evalúa con el 20 % restante.")

# ------------------------------------------------------------------ 2. Resultados
with t_res:
    st.markdown("Para cada pregunta probamos varios modelos y los comparamos con un **modelo base** que siempre responde "
                "el promedio. Un buen modelo debe ganarle con claridad.")
    st.subheader("💡 El hallazgo clave: el tamaño explica casi todo")
    c_t, c_i = st.columns([1.5, 1])
    with c_t:
        st.markdown(
            f"- El **área** de la finca explica casi toda la producción total (correlación **{R['corr_yield']['Farm_Area']:.3f}**).\n"
            f"- El **agua** también parece explicarla (**{R['corr_yield']['Water_Usage']:.3f}**), pero es una **correlación espuria**: "
            f"las fincas grandes usan más agua *y* producen más. Si quitamos el efecto del tamaño, la relación cae a "
            f"**{R['parcial_agua_yield']:.3f}**.\n"
            "- Por eso predecimos el **rendimiento por acre** y luego lo multiplicamos por el área "
            "(como calcular un sueldo: pago por hora × horas).")
    with c_i:
        st.info("🍦 Ejemplo: los que venden más helados también venden más bloqueador. No es que uno cause el otro: "
                "los dos los causa el verano. Aquí el “verano” es el **tamaño de la finca**.")

    modelos_img = {"Regresión lineal": ("lineal", "Regresión lineal: una relación en línea recta, como una carretera."),
                   "Random Forest": ("bosque", "Random Forest: cientos de árboles de decisión que votan; se promedia."),
                   "XGBoost": ("boost", "XGBoost: árboles en fila; cada uno corrige al anterior, rápido como un guepardo.")}
    dims = [("D1", "1️⃣ Productividad · toneladas totales", "R2", "Lineal y XGBoost predicen muy bien; la versión "
             f"“por acre × área” sube a R² {MJ['R2']:.3f} (error medio ±{MJ['MAE']:.0f} t).", "XGBoost"),
            ("D2", "2️⃣ Agua · m³ usados", "R2", "La relación es casi una línea recta con el área: un modelo lineal "
             "(Lasso) basta y es más fácil de explicar.", "Regresión lineal"),
            ("D3", "3️⃣ Insumos · toneladas por acre", "R2", "La regresión lineal gana; el fertilizante aporta poco frente "
             "al cultivo y al riego.", "Regresión lineal"),
            ("D4", "4️⃣ Cultivo y suelo · ¿productividad alta?", "AUC", "XGBoost separa mejor las fincas de alta y baja "
             "productividad; arroz y caña superan el 90 %.", "XGBoost")]
    for d, titulo, met, lectura, ganador in dims:
        with st.expander(titulo, expanded=(d == "D1")):
            m = D[d]["modelos"]
            s = pd.Series({n: v[met] for n, v in m.items()})
            if d == "D1":
                s["XGBoost (por acre × área)"] = MJ["R2"]
                s["Random Forest (por acre × área)"] = R["d1_mejorado"]["Random Forest (por acre × área)"]["R2"]
            s = s.sort_values(ascending=False).round(3)
            c_g, c_m = st.columns([1.7, 1])
            with c_g:
                barras(s, "R² en el 20 % de prueba" if met == "R2" else "AUC en el 20 % de prueba", resaltar=s.index[0])
                st.markdown(f"**Cómo leerlo:** barra más larga = modelo que acierta más. "
                            f"{'R² = 1 sería perfecto; 0 es lo mismo que adivinar el promedio.' if met == 'R2' else 'AUC = 1 sería perfecto; 0.5 es lanzar una moneda.'}")
                st.success(lectura)
            with c_m:
                img, cap = modelos_img[ganador]
                st.image(IMG[img], caption=cap, use_container_width=True)

# ------------------------------------------------------------------ 3. Escenarios
with t_esc:
    c_t, c_i = st.columns([1.5, 1])
    with c_t:
        st.subheader("¿Qué pasaría si…? Cuatro decisiones de manejo")
        st.markdown(
            "Cambiamos **una decisión** en las 600 fincas, el modelo entrenado vuelve a predecir y comparamos con la "
            "situación actual (escenarios **contrafactuales**). El agua se mantiene igual.")
        st.markdown(
            f"- **S1 · Riego tecnificado:** las {R['n_convertidas_riego']} fincas de secano o riego manual pasan a goteo.\n"
            "- **S2 · Más fertilizante:** todas usan 20 % más.\n"
            "- **S3 · Temporada óptima:** cada finca siembra en la temporada en la que más rinde.\n"
            "- **S4 · Plan combinado:** S1 + S3.")
    with c_i:
        st.image(IMG["riego"], caption="El riego por goteo es la palanca más fuerte.", use_container_width=True)

    st.subheader("Resultado del plan recomendado (S4) frente a hoy")
    c = st.columns(4)
    kpi(c[0], "Producción total", f"{S4[K[0]]:,.0f} t", f"{vp(PX.index[-1], K[0]):+.1f} % vs. {ACT[K[0]]:,.0f} t")
    kpi(c[1], "Rendimiento medio", f"{S4[K[1]]:.3f} t/acre", f"{vp(PX.index[-1], K[1]):+.1f} %")
    kpi(c[2], "Fincas de productividad alta", f"{S4[K[2]]:.1f} %", f"{S4[K[2]] - ACT[K[2]]:+.1f} pp vs. {ACT[K[2]]:.1f} %")
    kpi(c[3], "Productividad hídrica", f"{S4[K[3]]:.2f}", f"{vp(PX.index[-1], K[3]):+.1f} % (misma agua)")

    st.subheader("Compare los escenarios")
    ind = st.selectbox("Elija el indicador (KPI) que quiere mejorar", K,
                       help="KPI = indicador clave. Producción total (t), rendimiento por acre, % de fincas de productividad alta "
                            "y toneladas por cada 1,000 m³ de agua.")
    var = (PX[ind] / ACT[ind] - 1) * 100
    c1, c2 = st.columns([1.25, 1])
    c1.dataframe(pd.DataFrame({"Escenario": PX.index, "Valor": PX[ind].round(2 if PX[ind].max() < 1000 else 0),
                               "Var. vs. hoy (%)": var.round(1)}),
                 hide_index=True, use_container_width=True,
                 column_config={"Escenario": st.column_config.TextColumn(width="medium")})
    with c2:
        barras(var.iloc[1:].round(1), "Variación vs. hoy (%)", resaltar=var.iloc[1:].idxmax(), fmt="+.1f", orden=list(var.index[1:]))
    mejor = var.iloc[1:].idxmax()
    st.success(f"✅ Para «{ind}», el mejor escenario es **{mejor}** ({var[mejor]:+.1f} %). "
               f"Subir el fertilizante (S2) solo aporta {var.iloc[2]:+.1f} %.")
    st.warning("⚠️ Limitación: el dataset es sintético y no incluye clima. Antes de aplicar el plan conviene un piloto de campo.")

    with st.expander("📈 Escenarios pesimista / base / optimista por dimensión"):
        txt = {"D1": "Riego: todo secano / actual / todo goteo (producción total)",
               "D2": "Riego: productividad hídrica (t por 1,000 m³)",
               "D3": "Fertilizante −20 % y pesticida +20 % / actual / al revés (t/acre)",
               "D4": "Temporada: peor / actual / mejor (fincas de productividad alta)"}
        filas = []
        for d in ["D1", "D2", "D3", "D4"]:
            for n in ["Random Forest", "XGBoost"]:
                e = R["escenarios"][n][d]
                filas.append({"Dimensión": d, "Supuesto": txt[d], "Modelo": n,
                              "Pesimista": f"{(e['Pesimista']/e['Base']-1)*100:+.1f} %",
                              "Optimista": f"{(e['Optimista']/e['Base']-1)*100:+.1f} %"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 4. Glosario
with t_glo:
    st.subheader("Términos en una línea")
    g = [("R²", "Qué tanto acierta un modelo que predice números. 1 = perfecto; 0 = igual que adivinar el promedio."),
         ("AUC", "Qué tan bien separa un modelo los “sí” de los “no”. 1 = perfecto; 0.5 = lanzar una moneda."),
         ("Modelo base", "Un modelo tonto que siempre responde el promedio. Sirve de referencia mínima."),
         ("80/20 y validación cruzada", "Se entrena con el 80 % de fincas y se evalúa con el 20 % que el modelo no vio; "
          "se repite 5 veces para ver si el resultado es estable."),
         ("Regresión lineal, Ridge y Lasso", "Una línea recta que relaciona variables con el resultado. Ridge y Lasso "
          "le ponen freno; Lasso además descarta las variables que no aportan."),
         ("Regresión logística", "Como la lineal, pero responde con una probabilidad (por ejemplo, 80 % de ser productividad alta)."),
         ("Random Forest", "Cientos de árboles de decisión que votan; se promedia su respuesta."),
         ("XGBoost", "Árboles en fila: cada uno corrige el error del anterior."),
         ("PCA", "Junta variables que dicen lo mismo (como área y agua) en menos columnas."),
         ("VIF", "Mide si dos variables se repiten; más de 10 indica que una sobra."),
         ("Correlación espuria", "Dos cosas parecen relacionadas, pero las mueve una tercera (aquí, el tamaño de la finca)."),
         ("Escenario contrafactual", "Cambiar una decisión en los datos y ver qué predice el modelo."),
         ("Prescribir", "Recomendar la mejor decisión, no solo predecir.")]
    c1, c2 = st.columns(2)
    for i, (t, x) in enumerate(g):
        with (c1 if i % 2 == 0 else c2):
            tarjeta(t, x)

st.divider()
st.caption("Fuente de las cifras: analisis_final.py (results.json). Imágenes: Unsplash (licencia libre).")
