# App Grupo 3 · Predicción del rendimiento agrícola

Modelamiento Predictivo de Datos · Universidad de Lima · Bajalqui, De La Cuba, Perdicci

## Ejecutar en la computadora
```
pip install -r requirements.txt
streamlit run Inicio.py
```

## Páginas
- Inicio · 0 Regresión lineal · 1 Random Forest · 2 XGBoost · 3 Reducción de dimensiones (estilo.py tiene el diseño y los textos de ayuda)

## Publicar gratis en Streamlit Community Cloud
1. Crear una cuenta en https://github.com y un repositorio **público** nuevo (por ejemplo `grupo3-rendimiento-agricola`).
2. En el repositorio: **Add file → Upload files** y arrastrar TODO el contenido de esta carpeta
   (Inicio.py, la carpeta pages/, agriculture_dataset_synthetic.csv, results.json, requirements.txt, README.md) → **Commit changes**.
3. Entrar a https://share.streamlit.io e iniciar sesión con GitHub.
4. **Create app → Deploy a public app from GitHub**: Repository = el repositorio, Branch = `main`, Main file path = `Inicio.py`.
5. (Opcional) **App URL**: escribir un nombre corto, por ejemplo `grupo3-agricola`.
6. **Deploy**. En 2–4 minutos queda en `https://<nombre>.streamlit.app`. Ese enlace va en el póster.

Si sale "ModuleNotFoundError", revisar que requirements.txt esté en la raíz del repositorio.
