# -*- coding: utf-8 -*-
"""Estilo visual común de la app (banner con imagen, tarjetas y guía de uso).

Las imágenes son de Unsplash (licencia libre) y se cargan desde su URL, igual que en la app de la EC1.
"""
import re

import streamlit as st

IMG = {
    "hero": "https://images.unsplash.com/photo-1560493676-04071c5f467b?w=1600&q=80&auto=format&fit=crop",
    "riego": "https://images.unsplash.com/photo-1692369584496-3216a88f94c1?w=900&q=80&auto=format&fit=crop",
    "trigo": "https://images.unsplash.com/photo-1753081490091-37ce36d8bbd8?w=900&q=80&auto=format&fit=crop",
    "suelo": "https://images.unsplash.com/photo-1457530378978-8bac673b8062?w=900&q=80&auto=format&fit=crop",
    "lineal": "https://images.unsplash.com/photo-1749030921691-ca25b4116921?w=900&q=80&auto=format&fit=crop",   # carretera recta
    "bosque": "https://images.unsplash.com/photo-1643400988715-6a42d42c1b86?w=900&q=80&auto=format&fit=crop",   # bosque
    "boost": "https://images.unsplash.com/photo-1498056523130-b76aa40137b0?w=900&q=80&auto=format&fit=crop",    # guepardo
}

_CSS = """
<style>
.hero {background: linear-gradient(rgba(60,20,5,.55), rgba(60,20,5,.85)), url('%s');
       background-size: cover; background-position: center 60%%; border-radius: 18px;
       padding: 2.4rem 2.2rem; margin: .2rem 0 1.2rem;}
.hero h1 {color:#fff !important; font-size:2.25rem; margin:0 0 .4rem; line-height:1.15;}
.hero p {color:#FBE3D2 !important; font-size:1.05rem; margin:0; max-width:900px;}
.hero .pill {display:inline-block; background:#FFD58A; color:#5a1f08; font-weight:700; border-radius:30px;
             padding:.25rem .9rem; margin-top:.8rem; font-size:.95rem;}
.card {background:#FDF3EA; border-left:6px solid #C0562A; border-radius:12px; padding:.9rem 1.1rem; margin-bottom:.8rem;}
.card h4 {margin:0 0 .3rem; color:#7a2e10;}
.card p {margin:0; color:#33302c;}
.kpi {background:linear-gradient(135deg,#7a2e10,#C0562A); border-radius:14px; padding:1rem 1.1rem; color:#fff;}
.kpi .l {font-size:.85rem; color:#FBE3D2; font-weight:600;}
.kpi .n {font-size:1.9rem; font-weight:800; line-height:1.1;}
.simple {background:#EEF5FF; border-left:6px solid #2a78d6; border-radius:10px; padding:.7rem 1rem; margin:.4rem 0 1rem; color:#1d2b3a;}
.simple b.t {color:#1d4ed8;}
.kpi .d {display:inline-block; background:#FFD58A; color:#5a1f08; font-weight:700; border-radius:20px;
         padding:.05rem .6rem; font-size:.8rem; margin-top:.3rem;}
</style>
"""


def hero(titulo, texto, img="hero", pill=None):
    """Banner superior con imagen de fondo, título y una frase que resume la página."""
    st.markdown(_CSS % IMG.get(img, img), unsafe_allow_html=True)
    extra = f"<div class='pill'>{pill}</div>" if pill else ""
    st.markdown(f"<div class='hero'><h1>{titulo}</h1><p>{texto}</p>{extra}</div>", unsafe_allow_html=True)


def guia(pasos, titulo="🧭 ¿Cómo usar esta página?"):
    """Caja desplegable con los pasos para leer o usar la página."""
    with st.expander(titulo, expanded=False):
        for i, p in enumerate(pasos, 1):
            st.markdown(f"**{i}.** {p}")


def tarjeta(titulo, texto):
    texto = _md(texto)
    st.markdown(f"<div class='card'><h4>{titulo}</h4><p>{texto}</p></div>", unsafe_allow_html=True)


def kpi(col, etiqueta, valor, delta=None):
    d = f"<div class='d'>{delta}</div>" if delta else ""
    col.markdown(f"<div class='kpi'><div class='l'>{etiqueta}</div><div class='n'>{valor}</div>{d}</div>", unsafe_allow_html=True)


def _md(t):
    """Convierte **negrita** y *cursiva* de markdown a HTML (dentro de bloques HTML no se interpretan)."""
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)\*(?![\w*])", r"<i>\1</i>", t)


def explica(texto):
    """Recuadro azul 'En palabras simples' que va junto a cada gráfico o tabla."""
    texto = _md(texto)
    st.markdown(f"<div class='simple'><b class='t'>💬 En palabras simples:</b> {texto}</div>", unsafe_allow_html=True)


# Textos de los globos de ayuda (aparecen al pasar el cursor por el ícono ? de cada control o métrica)
AYUDA = {
    "R2": "R² (de 0 a 1): qué parte de las diferencias entre fincas explica el modelo. 0.95 = explica el 95 %. "
          "1 sería perfecto; 0 es lo mismo que adivinar siempre el promedio.",
    "RMSE": "Error típico de la predicción, en las mismas unidades (t, m³…). Castiga más los errores grandes. Menor es mejor.",
    "MAE": "Error promedio: en cuánto se equivoca el modelo, en promedio, por finca. Menor es mejor.",
    "AUC": "Qué tan bien separa el modelo las fincas de productividad alta de las de baja. 1 = perfecto; 0.5 = lanzar una moneda.",
    "Accuracy": "Porcentaje de fincas bien clasificadas (alta o baja).",
    "F1": "Equilibrio entre no dejar pasar fincas de productividad alta y no marcar como alta a las que no lo son. 1 = perfecto.",
    "train": "R² con los datos con los que el modelo aprendió. Si es mucho mayor que el de prueba, el modelo memorizó (sobreajuste).",
    "VIF": "Mide si una variable repite lo que dicen las otras. Más de 10 = redundante (aquí: área y agua).",
    "corr": "Correlación (de −1 a 1): qué tanto suben o bajan juntas dos variables. Cerca de 1 = muy relacionadas. "
            "Ojo: relación no siempre es causa.",
    "n_estimators": "Cantidad de árboles. Más árboles = resultado más estable, pero tarda más.",
    "max_depth": "Cuántas preguntas seguidas puede hacer cada árbol. Más profundo = más detalle, pero riesgo de memorizar.",
    "min_samples_leaf": "Mínimo de fincas que debe haber en cada grupo final del árbol. Más alto = modelo más simple.",
    "learning_rate": "Tamaño del paso con que cada árbol corrige al anterior. Pasos pequeños = aprende más lento pero más seguro.",
    "test_size": "Porcentaje de fincas que se guardan para examinar al modelo (no las ve al aprender). Usamos 20 %.",
    "pca_k": "Cuántas columnas-resumen (componentes) conservar. Con 3 se guarda casi el 99 % de la información.",
    "alpha": "Fuerza del freno (λ). Más alto = el modelo se simplifica más; Lasso empieza a apagar variables (coeficiente = 0).",
    "csv": "Opcional: puede subir otro archivo con las mismas columnas. Si no sube nada, se usan las 600 fincas del caso.",
    "nombre": "Texto que aparece en la franja verde de arriba (útil para las capturas).",
    "cultivo": "Tipo de cultivo de la finca simulada.", "suelo": "Tipo de suelo de la finca simulada.",
    "riego": "Sistema de riego: Drip = goteo, Sprinkler = aspersión, Flood = inundación, Manual = a mano, Rain-fed = solo lluvia.",
    "temporada": "Temporada de siembra: Kharif (lluvias, jun–oct), Rabi (invierno, nov–abr) y Zaid (verano corto, mar–jun).",
    "area": "Tamaño de la finca en acres (1 acre ≈ 0.4 hectáreas).",
    "fert": "Toneladas de fertilizante aplicadas en la finca.", "pest": "Kilogramos de pesticida aplicados en la finca.",
}
