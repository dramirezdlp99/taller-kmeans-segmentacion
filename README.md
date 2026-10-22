# 🛍️ Segmentación de Clientes con K-Means

Taller final de **Inteligencia de Negocios y Minería de Datos**.

| | |
|---|---|
| **Desarrollado por** | David Fernando Ramírez de la Parra |
| **Universidad** | Universidad Cooperativa de Colombia |
| **Facultad** | Facultad de Ingeniería de Software |
| **Materia** | Inteligencia de Negocios y Minería de Datos |
| **Profesor** | PhD Cristian Camilo Ordoñez Quintero |

## 🔗 Enlaces

- **Aplicación en línea (Streamlit):** https://segmentacion-clientes-kmeans-dramirez.streamlit.app/
- **Cuaderno en Google Colab:** https://colab.research.google.com/drive/1VqHgn5nD2hmIpVjh_fMCV6ZquNzoNExT?usp=sharing

## 📋 Descripción

Modelo de segmentación de clientes de un centro comercial usando el algoritmo **K-Means** sobre el dataset
[Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (200 clientes).
Se utilizaron las variables **ingreso anual** y **puntaje de gasto**, escaladas con `StandardScaler`.
El número óptimo de clusters (**K = 5**) se determinó con el **método del codo** y el **coeficiente de silueta** (≈ 0.555).

## 🎯 Perfiles obtenidos

| Cluster | Perfil | Clientes | Ingreso prom. (k$) | Puntaje prom. |
|---|---|---|---|---|
| 1 | 💎 Clientes de alto valor | 39 | 86.5 | 82.1 |
| 0 | 👥 Clientes promedio | 81 | 55.3 | 49.5 |
| 4 | 🏷️ Clientes de bajo valor | 23 | 26.3 | 20.9 |
| 3 | 🚀 Clientes con potencial | 35 | 88.2 | 17.1 |
| 2 | 🔥 Clientes jóvenes de alto gasto | 22 | 25.7 | 79.4 |

El segmento con **mayor oportunidad de crecimiento** es *Clientes con potencial*: tienen el ingreso más alto pero el gasto más bajo.

## 📁 Estructura del repositorio
├── data/Mall_Customers.csv # Dataset
├── taller_kmeans.ipynb # Cuaderno con el análisis (Partes 1 a 7) y respuestas
├── app.py # Aplicación Streamlit (Parte 7)
├── requirements.txt # Librerías necesarias
└── .streamlit/config.toml # Tema visual de la app


## ▶️ Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```