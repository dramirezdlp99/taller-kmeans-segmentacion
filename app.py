import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de Clientes | K-Means", page_icon="🛍️", layout="wide")

AUTOR = "David Fernando Ramírez de la Parra"
UNIVERSIDAD = "Universidad Cooperativa de Colombia"
FACULTAD = "Facultad de Ingeniería de Software"
MATERIA = "Inteligencia de Negocios y Minería de Datos"
PROFESOR = "PhD Cristian Camilo Ordoñez Quintero"
REPO = "https://github.com/dramirezdlp99/taller-kmeans-segmentacion"

VARIABLES = ["Ingreso", "Puntaje"]
K_OPTIMO = 5

PERFILES = {
    "Clientes de alto valor": {
        "color": "#22C55E",
        "icono": "💎",
        "descripcion": "Ingreso alto y gasto alto. Son los clientes más rentables del centro comercial.",
        "objetivo": "Retener y fidelizar",
        "estrategias": [
            "Programa VIP con beneficios exclusivos y preventas.",
            "Atención personalizada y eventos privados.",
            "Alianzas con marcas premium.",
        ],
    },
    "Clientes con potencial": {
        "color": "#3B82F6",
        "icono": "🚀",
        "descripcion": "Ingreso alto pero gasto bajo. Tienen capacidad de compra que hoy no usan en el centro comercial.",
        "objetivo": "Atraer y activar",
        "estrategias": [
            "Campañas personalizadas con tiendas y productos de gama alta.",
            "Experiencias: gastronomía, entretenimiento y eventos.",
            "Beneficios de bienvenida y encuestas para conocer qué buscan.",
        ],
    },
    "Clientes promedio": {
        "color": "#A1A1AA",
        "icono": "👥",
        "descripcion": "Ingreso y gasto medios. Es el grupo más grande y estable.",
        "objetivo": "Aumentar frecuencia y ticket",
        "estrategias": [
            "Programa de puntos y promociones periódicas.",
            "Combos y venta cruzada.",
            "Ofertas por temporada y recordatorios.",
        ],
    },
    "Clientes jóvenes de alto gasto": {
        "color": "#F59E0B",
        "icono": "🔥",
        "descripcion": "Ingreso bajo pero gasto alto. Son los más jóvenes y siguen tendencias.",
        "objetivo": "Mantener el entusiasmo",
        "estrategias": [
            "Marketing en redes sociales e influenciadores.",
            "Lanzamientos, moda y tendencias.",
            "Descuentos para estudiantes y facilidades de pago responsables.",
        ],
    },
    "Clientes de bajo valor": {
        "color": "#EF4444",
        "icono": "🏷️",
        "descripcion": "Ingreso bajo y gasto bajo. Su aporte actual es reducido.",
        "objetivo": "Atender con bajo costo",
        "estrategias": [
            "Ofertas de precio y descuentos en productos básicos.",
            "Días de promoción.",
            "Comunicación por canales económicos (correo, redes, WhatsApp).",
        ],
    },
}
ORDEN = list(PERFILES.keys())
COLORES = {p: info["color"] for p, info in PERFILES.items()}

CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg, #6366F1 0%, #8B5CF6 45%, #EC4899 100%); padding: 1.8rem 2rem; border-radius: 20px; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(139, 92, 246, 0.25);}
.hero h1 {color: #FFFFFF; margin: 0; font-size: 2.1rem;}
.hero p {color: #F5F3FF; margin: 0.4rem 0 0 0; font-size: 1.02rem;}
.chips {margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.5rem;}
.chips span {background: rgba(255, 255, 255, 0.18); border: 1px solid rgba(255, 255, 255, 0.35); color: #FFFFFF; padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600;}
div[data-testid="stMetric"] {background: #1E293B; border: 1px solid #334155; border-radius: 14px; padding: 0.9rem 1.1rem;}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
.stTabs [data-baseweb="tab"] {background: #1E293B; border-radius: 10px 10px 0 0; padding: 0.55rem 1rem;}
.stTabs [data-baseweb="tab"] p {font-size: 0.98rem; font-weight: 600;}
.card {background: #1E293B; border: 1px solid #334155; border-radius: 14px; padding: 1rem 1.1rem; height: 100%; margin-bottom: 0.8rem;}
.card h4 {margin: 0.2rem 0 0.4rem 0; font-size: 1.02rem; color: #F1F5F9;}
.card p, .card li {font-size: 0.92rem; color: #CBD5E1; margin: 0.15rem 0;}
.perfil {border-left: 6px solid var(--c);}
.etiqueta {display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; background: var(--c); color: #0F172A;}
.resultado {border-radius: 18px; padding: 1.4rem 1.5rem; border: 2px solid var(--c); background: #1E293B;}
.resultado h2 {margin: 0.3rem 0; color: var(--c);}
.resultado p, .resultado li {color: #CBD5E1;}
.ficha {background: linear-gradient(160deg, rgba(99, 102, 241, 0.18), rgba(236, 72, 153, 0.12)); border: 1px solid #475569; border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 0.6rem;}
.ficha .t {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #A5B4FC; margin: 0.55rem 0 0 0; font-weight: 700;}
.ficha .t:first-child {margin-top: 0;}
.ficha .v {font-size: 0.92rem; color: #F1F5F9; margin: 0.1rem 0 0 0; font-weight: 600; line-height: 1.3;}
.footer {margin-top: 2.5rem; padding: 1.2rem 1rem; border-top: 1px solid #334155; text-align: center; color: #94A3B8; font-size: 0.88rem; line-height: 1.7;}
.footer b {color: #E2E8F0;}
.footer a {color: #A5B4FC; text-decoration: none;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/Mall_Customers.csv")
    df = df.rename(columns={
        "Genre": "Genero",
        "Gender": "Genero",
        "Age": "Edad",
        "Annual Income (k$)": "Ingreso",
        "Spending Score (1-100)": "Puntaje",
    })
    df = df.drop(columns=["CustomerID"]).dropna().drop_duplicates()
    df["Genero"] = df["Genero"].str.strip().str.capitalize()
    return df


def nombrar_clusters(centroides_esc):
    perfiles = {}
    distancias = np.linalg.norm(centroides_esc, axis=1)
    c_promedio = int(np.argmin(distancias))
    for i, (ing, pun) in enumerate(centroides_esc):
        if i == c_promedio:
            perfiles[i] = "Clientes promedio"
        elif ing > 0 and pun > 0:
            perfiles[i] = "Clientes de alto valor"
        elif ing > 0 and pun <= 0:
            perfiles[i] = "Clientes con potencial"
        elif ing <= 0 and pun > 0:
            perfiles[i] = "Clientes jóvenes de alto gasto"
        else:
            perfiles[i] = "Clientes de bajo valor"
    return perfiles


@st.cache_resource
def entrenar():
    df = cargar_datos()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[VARIABLES])
    inercias = []
    siluetas = []
    for k in range(1, 11):
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
        inercias.append(modelo.inertia_)
        if k >= 2:
            siluetas.append(silhouette_score(X_scaled, modelo.labels_))
    kmeans = KMeans(n_clusters=K_OPTIMO, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
    return scaler, kmeans, X_scaled, inercias, siluetas


df = cargar_datos()
scaler, kmeans, X_scaled, inercias, siluetas = entrenar()

perfiles = nombrar_clusters(kmeans.cluster_centers_)
df["Cluster"] = kmeans.labels_
df["Perfil"] = df["Cluster"].map(perfiles)

centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=VARIABLES)
centroides["Cluster"] = range(K_OPTIMO)
centroides["Perfil"] = centroides["Cluster"].map(perfiles)

silueta = silhouette_score(X_scaled, kmeans.labels_)

resumen = df.groupby("Perfil").agg(
    Cluster=("Cluster", "first"),
    Clientes=("Ingreso", "count"),
    Edad=("Edad", "mean"),
    Ingreso=("Ingreso", "mean"),
    Puntaje=("Puntaje", "mean"),
).reindex(ORDEN).round(1)
resumen["% del total"] = (resumen["Clientes"] / len(df) * 100).round(1)


def grafico_clusters(nuevo=None, alto=520):
    fig = px.scatter(
        df, x="Ingreso", y="Puntaje", color="Perfil",
        color_discrete_map=COLORES, category_orders={"Perfil": ORDEN},
        hover_data={"Edad": True, "Genero": True, "Cluster": True},
        labels={"Ingreso": "Ingreso anual (k$)", "Puntaje": "Puntaje de gasto (1-100)"},
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="#0F172A")))
    fig.add_trace(go.Scatter(
        x=centroides["Ingreso"], y=centroides["Puntaje"], mode="markers", name="Centroides",
        marker=dict(symbol="x", size=16, color="#FFFFFF", line=dict(width=2, color="#0F172A")),
        text=centroides["Perfil"],
        hovertemplate="Centroide: %{text}<br>Ingreso: %{x:.1f} k$<br>Puntaje: %{y:.1f}<extra></extra>",
    ))
    if nuevo is not None:
        fig.add_trace(go.Scatter(
            x=[nuevo[0]], y=[nuevo[1]], mode="markers", name="Nuevo cliente",
            marker=dict(symbol="star", size=26, color="#FACC15", line=dict(width=2, color="#0F172A")),
            hovertemplate="Nuevo cliente<br>Ingreso: %{x:.1f} k$<br>Puntaje: %{y}<extra></extra>",
        ))
    fig.update_layout(height=alto, legend_title_text="Perfil", margin=dict(l=10, r=10, t=30, b=10))
    return fig


with st.sidebar:
    st.markdown("## 🛍️ Mall Customers")
    st.markdown("**Taller final · Segmentación con K-Means**")
    st.markdown(
        f"<div class='ficha'>"
        f"<p class='t'>👨‍💻 Desarrollado por</p><p class='v'>{AUTOR}</p>"
        f"<p class='t'>🎓 Universidad</p><p class='v'>{UNIVERSIDAD}</p>"
        f"<p class='t'>🏛️ Facultad</p><p class='v'>{FACULTAD}</p>"
        f"<p class='t'>📚 Materia</p><p class='v'>{MATERIA}</p>"
        f"<p class='t'>👨‍🏫 Profesor</p><p class='v'>{PROFESOR}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**⚙️ Ficha técnica**")
    st.markdown("**Algoritmo:** K-Means")
    st.markdown(f"**Clusters:** {K_OPTIMO}")
    st.markdown("**Variables:** ingreso anual y puntaje de gasto")
    st.markdown("**Escalado:** StandardScaler")
    st.markdown("**Validación:** método del codo y coeficiente de silueta")
    st.divider()
    st.markdown(f"[📂 Ver repositorio en GitHub]({REPO})")

st.markdown(
    f"<div class='hero'><h1>🛍️ Segmentación de Clientes con K-Means</h1>"
    f"<p>Análisis de 200 clientes de un centro comercial para identificar perfiles de consumo y apoyar decisiones de marketing.</p>"
    f"<div class='chips'>"
    f"<span>👨‍💻 {AUTOR}</span>"
    f"<span>🎓 {UNIVERSIDAD}</span>"
    f"<span>📚 {MATERIA}</span>"
    f"</div></div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("👥 Clientes analizados", len(df))
m2.metric("🎯 Clusters (K)", K_OPTIMO)
m3.metric("📐 Coef. de silueta", f"{silueta:.3f}")
m4.metric("📉 Inercia (WCSS)", f"{kmeans.inertia_:.2f}")

st.write("")

tab_inicio, tab_exp, tab_k, tab_clusters, tab_estr, tab_nuevo = st.tabs([
    "🏠 Resumen", "🔍 Exploración", "📉 Selección de K", "🎯 Clusters", "💼 Estrategias", "🧍 Clasificar cliente"
])

with tab_inicio:
    st.subheader("¿Qué hace esta aplicación?")
    st.markdown(
        "Agrupa a los clientes según su **ingreso anual** y su **puntaje de gasto** usando el algoritmo "
        "**K-Means**, y permite clasificar automáticamente a un cliente nuevo en el segmento al que pertenece."
    )
    pasos = [
        ("1️⃣ Exploración", "Revisión de variables, valores atípicos y relaciones entre ingreso, gasto y edad."),
        ("2️⃣ Preprocesamiento", "Limpieza de datos y escalado con StandardScaler (media 0, desviación 1)."),
        ("3️⃣ Selección de K", "Método del codo y coeficiente de silueta: ambos indican K = 5."),
        ("4️⃣ Modelo y negocio", "Entrenamiento de K-Means, perfiles de clientes y estrategias de marketing."),
    ]
    cols = st.columns(4)
    for col, (titulo, texto) in zip(cols, pasos):
        col.markdown(f"<div class='card'><h4>{titulo}</h4><p>{texto}</p></div>", unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        st.subheader("Hallazgos clave")
        pot = resumen.loc["Clientes con potencial"]
        prom = resumen.loc["Clientes promedio"]
        alto = resumen.loc["Clientes de alto valor"]
        st.markdown(f"- Se identificaron **{K_OPTIMO} segmentos** bien separados (silueta = {silueta:.3f}).")
        st.markdown(f"- **Clientes promedio** es el grupo más grande: {int(prom['Clientes'])} clientes ({prom['% del total']} %).")
        st.markdown(f"- **Clientes de alto valor** ({int(alto['Clientes'])} clientes) combinan ingreso de {alto['Ingreso']} k$ y gasto de {alto['Puntaje']}.")
        st.markdown(f"- **Clientes con potencial** tienen el ingreso más alto ({pot['Ingreso']} k$) pero el gasto más bajo ({pot['Puntaje']}): son la **mayor oportunidad de crecimiento**.")
        st.markdown("- Los segmentos de gasto alto son los más jóvenes; ingreso y gasto **no** están correlacionados.")
    with der:
        fig_pie = px.pie(
            resumen.reset_index(), names="Perfil", values="Clientes", color="Perfil",
            color_discrete_map=COLORES, hole=0.55,
        )
        fig_pie.update_traces(textinfo="percent")
        fig_pie.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Distribución de clientes por perfil")
        st.plotly_chart(fig_pie, width="stretch")

with tab_exp:
    st.subheader("Exploración de datos")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("**Primeros registros (después de la limpieza)**")
        st.dataframe(df[["Genero", "Edad", "Ingreso", "Puntaje"]].head(10), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Estadística descriptiva**")
        st.dataframe(df[["Edad", "Ingreso", "Puntaje"]].describe().round(2), width="stretch")

    st.markdown("**Distribución de las variables**")
    h1, h2, h3 = st.columns(3)
    for col, var in zip([h1, h2, h3], ["Edad", "Ingreso", "Puntaje"]):
        fig_h = px.histogram(df, x=var, nbins=20, marginal="box", color_discrete_sequence=["#8B5CF6"])
        fig_h.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title=var, showlegend=False)
        col.plotly_chart(fig_h, width="stretch")

    r1, r2 = st.columns(2, gap="large")
    with r1:
        corr = df[["Edad", "Ingreso", "Puntaje"]].corr().round(2)
        fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig_c.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), title="Matriz de correlación")
        st.plotly_chart(fig_c, width="stretch")
    with r2:
        fig_s = px.scatter(
            df, x="Ingreso", y="Puntaje", color_discrete_sequence=["#EC4899"],
            labels={"Ingreso": "Ingreso anual (k$)", "Puntaje": "Puntaje de gasto (1-100)"},
        )
        fig_s.update_traces(marker=dict(size=9, opacity=0.8))
        fig_s.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), title="Ingreso vs Puntaje (antes de agrupar)")
        st.plotly_chart(fig_s, width="stretch")

    st.info(
        "Solo la variable **Ingreso** presenta 2 valores atípicos (137 k$), que se conservaron por ser datos reales. "
        "La correlación Ingreso–Puntaje es casi 0, pero el diagrama de dispersión muestra cinco grupos naturales."
    )

with tab_k:
    st.subheader("Selección del número de clusters")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        fig_codo = go.Figure(go.Scatter(
            x=list(range(1, 11)), y=inercias, mode="lines+markers",
            line=dict(color="#6366F1", width=3), marker=dict(size=10),
            hovertemplate="K = %{x}<br>Inercia = %{y:.2f}<extra></extra>",
        ))
        fig_codo.add_vline(x=K_OPTIMO, line_dash="dash", line_color="#EF4444", annotation_text=f"K = {K_OPTIMO}")
        fig_codo.update_layout(height=400, title="Método del codo", xaxis_title="Número de clusters (K)",
                               yaxis_title="Inercia (WCSS)", margin=dict(l=10, r=10, t=50, b=10))
        fig_codo.update_xaxes(dtick=1)
        st.plotly_chart(fig_codo, width="stretch")
    with g2:
        fig_sil = go.Figure(go.Scatter(
            x=list(range(2, 11)), y=siluetas, mode="lines+markers",
            line=dict(color="#22C55E", width=3), marker=dict(size=10),
            hovertemplate="K = %{x}<br>Silueta = %{y:.3f}<extra></extra>",
        ))
        fig_sil.add_vline(x=K_OPTIMO, line_dash="dash", line_color="#EF4444", annotation_text=f"K = {K_OPTIMO}")
        fig_sil.update_layout(height=400, title="Coeficiente de silueta", xaxis_title="Número de clusters (K)",
                              yaxis_title="Silueta", margin=dict(l=10, r=10, t=50, b=10))
        fig_sil.update_xaxes(dtick=1)
        st.plotly_chart(fig_sil, width="stretch")

    tabla_k = pd.DataFrame({"K": range(1, 11), "Inercia": inercias, "Silueta": [np.nan] + siluetas}).round(3)
    t1, t2 = st.columns([2, 3], gap="large")
    with t1:
        st.dataframe(tabla_k, width="stretch", hide_index=True)
    with t2:
        st.success(
            f"**Conclusión:** la inercia deja de bajar con fuerza a partir de **K = {K_OPTIMO}** (codo) y la silueta "
            f"alcanza su máximo en **K = {K_OPTIMO}** ({silueta:.3f}). Ambos métodos coinciden, por lo que se eligen "
            f"{K_OPTIMO} clusters."
        )
        st.markdown("- **Inercia:** suma de distancias al cuadrado de cada cliente a su centroide (menor = grupos más compactos).")
        st.markdown("- **Silueta:** compara cercanía al propio grupo vs. al grupo vecino (cerca de 1 = grupos bien separados).")

with tab_clusters:
    st.subheader("Clusters generados")
    st.plotly_chart(grafico_clusters(), width="stretch")

    cols = st.columns(len(ORDEN))
    for col, perfil in zip(cols, ORDEN):
        if perfil not in resumen.index or pd.isna(resumen.loc[perfil, "Clientes"]):
            continue
        r = resumen.loc[perfil]
        color = PERFILES[perfil]["color"]
        icono = PERFILES[perfil]["icono"]
        col.markdown(
            f"<div class='card perfil' style='--c:{color}'>"
            f"<div style='font-size:1.8rem'>{icono}</div>"
            f"<h4>{perfil}</h4>"
            f"<span class='etiqueta'>Cluster {int(r['Cluster'])}</span>"
            f"<p style='margin-top:0.6rem'>👥 {int(r['Clientes'])} clientes ({r['% del total']} %)</p>"
            f"<p>💰 Ingreso: {r['Ingreso']} k$</p>"
            f"<p>🛒 Puntaje: {r['Puntaje']}</p>"
            f"<p>🎂 Edad: {r['Edad']} años</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("🧊 Ver los clusters en 3D (incluyendo la edad)"):
        fig_3d = px.scatter_3d(
            df, x="Ingreso", y="Puntaje", z="Edad", color="Perfil",
            color_discrete_map=COLORES, category_orders={"Perfil": ORDEN}, opacity=0.85,
        )
        fig_3d.update_traces(marker=dict(size=5))
        fig_3d.update_layout(height=560, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_3d, width="stretch")
        st.caption("La edad no se usó para agrupar, pero ayuda a describir los segmentos: los de gasto alto son más jóvenes.")

with tab_estr:
    st.subheader("Estrategias de marketing por perfil")
    pot = resumen.loc["Clientes con potencial"]
    st.success(
        f"🚀 **Mayor oportunidad de crecimiento: Clientes con potencial.** Tienen el ingreso promedio más alto "
        f"({pot['Ingreso']} k$) pero el gasto más bajo ({pot['Puntaje']}). Ya tienen capacidad de compra; "
        f"el reto es lograr que la usen en el centro comercial."
    )
    for i in range(0, len(ORDEN), 2):
        fila = st.columns(2, gap="large")
        for col, perfil in zip(fila, ORDEN[i:i + 2]):
            info = PERFILES[perfil]
            color = info["color"]
            lista = "".join(f"<li>{x}</li>" for x in info["estrategias"])
            col.markdown(
                f"<div class='card perfil' style='--c:{color}'>"
                f"<h4>{info['icono']} {perfil}</h4>"
                f"<span class='etiqueta'>{info['objetivo']}</span>"
                f"<p style='margin-top:0.6rem'>{info['descripcion']}</p>"
                f"<ul>{lista}</ul>"
                f"</div>",
                unsafe_allow_html=True,
            )

with tab_nuevo:
    st.subheader("Clasificar un nuevo cliente")
    st.caption("El cliente se escala con el mismo StandardScaler del entrenamiento y se asigna al centroide más cercano.")
    izq, der = st.columns([1, 2], gap="large")

    with izq:
        with st.form("form_cliente"):
            ingreso = st.number_input("💰 Ingreso anual (k$)", min_value=0.0, max_value=200.0, value=60.0, step=1.0)
            puntaje = st.slider("🛒 Puntaje de gasto (1-100)", min_value=1, max_value=100, value=50)
            enviar = st.form_submit_button("Clasificar cliente", type="primary")

    with der:
        if enviar:
            nuevo = pd.DataFrame([[ingreso, puntaje]], columns=VARIABLES)
            nuevo_esc = scaler.transform(nuevo)
            cluster = int(kmeans.predict(nuevo_esc)[0])
            perfil = perfiles[cluster]
            info = PERFILES[perfil]
            color = info["color"]
            lista = "".join(f"<li>{x}</li>" for x in info["estrategias"])
            st.markdown(
                f"<div class='resultado' style='--c:{color}'>"
                f"<div style='font-size:2.4rem'>{info['icono']}</div>"
                f"<h2>{perfil}</h2>"
                f"<span class='etiqueta'>Cluster {cluster}</span>"
                f"<p style='margin-top:0.7rem'>{info['descripcion']}</p>"
                f"<p><b>Objetivo:</b> {info['objetivo']}</p>"
                f"<ul>{lista}</ul>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("👈 Ingresa los datos del cliente y presiona **Clasificar cliente**.")

    if enviar:
        st.write("")
        b1, b2 = st.columns(2, gap="large")
        with b1:
            cent = centroides.loc[cluster]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name="Nuevo cliente", x=["Ingreso (k$)", "Puntaje"], y=[ingreso, puntaje], marker_color="#FACC15"))
            fig_comp.add_trace(go.Bar(name="Centroide de su cluster", x=["Ingreso (k$)", "Puntaje"], y=[cent["Ingreso"], cent["Puntaje"]], marker_color=color))
            fig_comp.add_trace(go.Bar(name="Promedio general", x=["Ingreso (k$)", "Puntaje"], y=[df["Ingreso"].mean(), df["Puntaje"].mean()], marker_color="#64748B"))
            fig_comp.update_layout(barmode="group", height=360, title="Comparación con su segmento", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_comp, width="stretch")
        with b2:
            dist = np.linalg.norm(kmeans.cluster_centers_ - nuevo_esc, axis=1)
            tabla_dist = pd.DataFrame({"Perfil": [perfiles[i] for i in range(K_OPTIMO)], "Distancia": dist}).sort_values("Distancia")
            fig_d = px.bar(tabla_dist, x="Distancia", y="Perfil", orientation="h", color="Perfil", color_discrete_map=COLORES)
            fig_d.update_layout(height=360, title="Distancia a cada centroide (escala estandarizada)", showlegend=False,
                                yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_d, width="stretch")
        st.plotly_chart(grafico_clusters(nuevo=(ingreso, puntaje), alto=480), width="stretch")

st.markdown(
    f"<div class='footer'>"
    f"Desarrollado por <b>{AUTOR}</b><br>"
    f"{FACULTAD} · {UNIVERSIDAD}<br>"
    f"{MATERIA} · Profesor: <b>{PROFESOR}</b><br>"
    f"<a href='{REPO}' target='_blank'>📂 Repositorio en GitHub</a> · 2026"
    f"</div>",
    unsafe_allow_html=True,
)