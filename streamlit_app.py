# streamlit_app.py — Unsupervised Learning Explorer
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Unsupervised Learning Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0a0e1a; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2a3a;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2235 100%);
    border: 1px solid #2a3a50;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #4fd1c5;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .lbl {
    font-size: 0.8rem;
    color: #7a8fa6;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Algorithm explanation box */
.algo-box {
    background: linear-gradient(135deg, #1a1f35 0%, #1e2840 100%);
    border-left: 4px solid #4fd1c5;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.92rem;
    color: #c8d6e5;
    line-height: 1.6;
}
.algo-box strong { color: #4fd1c5; }

/* Topic pill */
.topic-pill {
    display: inline-block;
    background: linear-gradient(90deg, #4fd1c5, #3b82f6);
    color: #0a0e1a;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* Page title */
.page-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4fd1c5, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 4px;
}
.page-subtitle {
    color: #7a8fa6;
    font-size: 1rem;
    margin-bottom: 24px;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    border-bottom: 2px solid #2a3a50;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #1a2a1a 0%, #1f2f1f 100%);
    border: 1px solid #2d5a2d;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #90ee90;
    font-size: 0.9rem;
}
.insight-box.warning {
    background: linear-gradient(135deg, #2a2010 0%, #2f2515 100%);
    border-color: #7a5a10;
    color: #ffd700;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
IRIS_CSV = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
4.8,3.4,1.6,0.2,setosa
4.8,3.0,1.4,0.1,setosa
4.3,3.0,1.1,0.1,setosa
5.8,4.0,1.2,0.2,setosa
5.7,4.4,1.5,0.4,setosa
5.4,3.9,1.3,0.4,setosa
5.1,3.5,1.4,0.3,setosa
5.7,3.8,1.7,0.3,setosa
5.1,3.8,1.5,0.3,setosa
5.4,3.4,1.7,0.2,setosa
5.1,3.7,1.5,0.4,setosa
4.6,3.6,1.0,0.2,setosa
5.1,3.3,1.7,0.5,setosa
4.8,3.4,1.9,0.2,setosa
5.0,3.0,1.6,0.2,setosa
5.0,3.4,1.6,0.4,setosa
5.2,3.5,1.5,0.2,setosa
5.2,3.4,1.4,0.2,setosa
4.7,3.2,1.6,0.2,setosa
4.8,3.1,1.6,0.2,setosa
5.4,3.4,1.5,0.4,setosa
5.2,4.1,1.5,0.1,setosa
5.5,4.2,1.4,0.2,setosa
4.9,3.1,1.5,0.2,setosa
5.0,3.2,1.2,0.2,setosa
5.5,3.5,1.3,0.2,setosa
4.9,3.6,1.4,0.1,setosa
4.4,3.0,1.3,0.2,setosa
5.1,3.4,1.5,0.2,setosa
5.0,3.5,1.3,0.3,setosa
4.5,2.3,1.3,0.3,setosa
4.4,3.2,1.3,0.2,setosa
5.0,3.5,1.6,0.6,setosa
5.1,3.8,1.9,0.4,setosa
4.8,3.0,1.4,0.3,setosa
5.1,3.8,1.6,0.2,setosa
4.6,3.2,1.4,0.2,setosa
5.3,3.7,1.5,0.2,setosa
5.0,3.3,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
5.0,2.0,3.5,1.0,versicolor
5.9,3.0,4.2,1.5,versicolor
6.0,2.2,4.0,1.0,versicolor
6.1,2.9,4.7,1.4,versicolor
5.6,2.9,3.6,1.3,versicolor
6.7,3.1,4.4,1.4,versicolor
5.6,3.0,4.5,1.5,versicolor
5.8,2.7,4.1,1.0,versicolor
6.2,2.2,4.5,1.5,versicolor
5.6,2.5,3.9,1.1,versicolor
5.9,3.2,4.8,1.8,versicolor
6.1,2.8,4.0,1.3,versicolor
6.3,2.5,4.9,1.5,versicolor
6.1,2.8,4.7,1.2,versicolor
6.4,2.9,4.3,1.3,versicolor
6.6,3.0,4.4,1.4,versicolor
6.8,2.8,4.8,1.4,versicolor
6.7,3.0,5.0,1.7,versicolor
6.0,2.9,4.5,1.5,versicolor
5.7,2.6,3.5,1.0,versicolor
5.5,2.4,3.8,1.1,versicolor
5.5,2.4,3.7,1.0,versicolor
5.8,2.7,3.9,1.2,versicolor
6.0,2.7,5.1,1.6,versicolor
5.4,3.0,4.5,1.5,versicolor
6.0,3.4,4.5,1.6,versicolor
6.7,3.1,4.7,1.5,versicolor
6.3,2.3,4.4,1.3,versicolor
5.6,3.0,4.1,1.3,versicolor
5.5,2.5,4.0,1.3,versicolor
5.5,2.6,4.4,1.2,versicolor
6.1,3.0,4.6,1.4,versicolor
5.8,2.6,4.0,1.2,versicolor
5.0,2.3,3.3,1.0,versicolor
5.6,2.7,4.2,1.3,versicolor
5.7,3.0,4.2,1.2,versicolor
5.7,2.9,4.2,1.3,versicolor
6.2,2.9,4.3,1.3,versicolor
5.1,2.5,3.0,1.1,versicolor
5.7,2.8,4.1,1.3,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
6.5,3.2,5.1,2.0,virginica
6.4,2.7,5.3,1.9,virginica
6.8,3.0,5.5,2.1,virginica
5.7,2.5,5.0,2.0,virginica
5.8,2.8,5.1,2.4,virginica
6.4,3.2,5.3,2.3,virginica
6.5,3.0,5.5,1.8,virginica
7.7,3.8,6.7,2.2,virginica
7.7,2.6,6.9,2.3,virginica
6.0,2.2,5.0,1.5,virginica
6.9,3.2,5.7,2.3,virginica
5.6,2.8,4.9,2.0,virginica
7.7,2.8,6.7,2.0,virginica
6.3,2.7,4.9,1.8,virginica
6.7,3.3,5.7,2.1,virginica
7.2,3.2,6.0,1.8,virginica
6.2,2.8,4.8,1.8,virginica
6.1,3.0,4.9,1.8,virginica
6.4,2.8,5.6,2.1,virginica
7.2,3.0,5.8,1.6,virginica
7.4,2.8,6.1,1.9,virginica
7.9,3.8,6.4,2.0,virginica
6.4,2.8,5.6,2.2,virginica
6.3,2.8,5.1,1.5,virginica
6.1,2.6,5.6,1.4,virginica
7.7,3.0,6.1,2.3,virginica
6.3,3.4,5.6,2.4,virginica
6.4,3.1,5.5,1.8,virginica
6.0,3.0,4.8,1.8,virginica
6.9,3.1,5.4,2.1,virginica
6.7,3.1,5.6,2.4,virginica
6.9,3.1,5.1,2.3,virginica
5.8,2.7,5.1,1.9,virginica
6.8,3.2,5.9,2.3,virginica
6.7,3.3,5.7,2.5,virginica
6.7,3.0,5.2,2.3,virginica
6.3,2.5,5.0,1.9,virginica
6.5,3.0,5.2,2.0,virginica
6.2,3.4,5.4,2.3,virginica
5.9,3.0,5.1,1.8,virginica"""

@st.cache_data
def get_iris():
    return pd.read_csv(StringIO(IRIS_CSV))

@st.cache_data
def get_mixed():
    try:
        return pd.read_csv("mixed_dataset.csv")
    except:
        return None

@st.cache_data
def get_movies():
    try:
        return pd.read_csv("Sample_Movies_Dataset.csv")
    except:
        return None

def preprocess(df, drop_cols=None):
    df2 = df.copy()
    if drop_cols:
        df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns], errors='ignore')
    for col in df2.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))
    df2 = df2.fillna(df2.median(numeric_only=True))
    scaler = StandardScaler()
    X = scaler.fit_transform(df2)
    return X, df2.columns.tolist()

def elbow_data(X, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(0)
    return list(k_range), inertias, silhouettes

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.5rem'>🧠</div>
        <div style='font-size:1.1rem; font-weight:700; color:#4fd1c5;'>Unsupervised<br>Learning Explorer</div>
        <div style='font-size:0.75rem; color:#7a8fa6; margin-top:6px;'>Interactive ML Toolkit</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📚 Topics")
    page = st.radio("", [
        "🏠 Overview",
        "📊 K-Means Clustering",
        "🌳 Hierarchical Clustering",
        "🔵 DBSCAN",
        "📉 PCA — Dimensionality Reduction",
        "🌀 t-SNE Visualization",
        "🎬 Movie Recommender",
        "👤 Customer Segmentation",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📂 Dataset")
    dataset_choice = st.selectbox("Choose dataset", ["Iris (built-in)", "Mixed Users", "Movies", "Upload CSV"])
    
    uploaded_file = None
    if dataset_choice == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset(choice, uploaded=None):
    if choice == "Iris (built-in)":
        return get_iris(), "species"
    elif choice == "Mixed Users":
        df = get_mixed()
        return df, None
    elif choice == "Movies":
        df = get_movies()
        return df, None
    elif choice == "Upload CSV" and uploaded is not None:
        return pd.read_csv(uploaded), None
    return get_iris(), "species"

df_raw, label_col = load_dataset(dataset_choice, uploaded_file)
if df_raw is None:
    st.error("Dataset not found. Please upload a CSV.")
    st.stop()

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="page-title">Unsupervised Learning Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Discover hidden patterns in data — no labels required</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="val">6</div><div class="lbl">ML Topics</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val">{len(df_raw):,}</div><div class="lbl">Data Rows</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="val">{len(df_raw.columns)}</div><div class="lbl">Features</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="val">3</div><div class="lbl">Datasets</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">What is Unsupervised Learning?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
        <strong>Unsupervised learning</strong> finds hidden structure in unlabelled data. Unlike supervised learning, 
        there's no "right answer" to learn from — the algorithm must discover patterns on its own.<br><br>
        Think of it like sorting a pile of foreign coins without knowing the country names. 
        You'd group them by color, size, and markings — that's clustering!
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Topics Covered in This App</div>', unsafe_allow_html=True)
    
    topics = [
        ("📊", "K-Means Clustering", "Partitions data into K groups by minimizing within-cluster distance. Best for spherical clusters."),
        ("🌳", "Hierarchical Clustering", "Builds a tree of clusters (dendrogram). No need to specify K upfront. Ward linkage works best."),
        ("🔵", "DBSCAN", "Density-Based Spatial Clustering. Finds clusters of arbitrary shape and detects outliers automatically."),
        ("📉", "PCA", "Principal Component Analysis reduces dimensions while preserving maximum variance. Great for visualization."),
        ("🌀", "t-SNE", "t-distributed Stochastic Neighbor Embedding. Best 2D visualization of high-dimensional clusters."),
        ("🎬", "Recommender System", "Uses clustering to group similar users/items and recommend based on cluster peers."),
    ]

    for i in range(0, len(topics), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(topics):
                icon, name, desc = topics[i + j]
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left; padding:18px;">
                        <div style="font-size:1.8rem">{icon}</div>
                        <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin:8px 0 6px 0;">{name}</div>
                        <div style="font-size:0.82rem; color:#7a8fa6; line-height:1.5;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Current Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data Types**")
        dtype_df = pd.DataFrame({"Column": df_raw.columns, "Type": df_raw.dtypes.astype(str), "Non-Null": df_raw.notnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True)
    with col2:
        st.markdown("**Numeric Statistics**")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: K-MEANS
# ─────────────────────────────────────────────
elif page == "📊 K-Means Clustering":
    st.markdown('<div class="topic-pill">Partitional Clustering</div>', unsafe_allow_html=True)
    st.markdown("## 📊 K-Means Clustering")
    
    st.markdown("""
    <div class="algo-box">
        <strong>How it works:</strong> K-Means randomly places K centroids, assigns every point to the nearest one, 
        then moves each centroid to the mean of its assigned points. Repeats until stable.<br><br>
        <strong>Best for:</strong> Compact, spherical clusters of similar sizes.<br>
        <strong>Key parameter:</strong> K (number of clusters) — use the Elbow Method or Silhouette Score to choose.
    </div>
    """, unsafe_allow_html=True)

    # Controls
    col1, col2 = st.columns([1, 2])
    with col1:
        label_cols_km = [c for c in df_raw.columns if df_raw[c].dtype == 'object' or c == label_col]
        drop_km = st.multiselect("Columns to exclude (labels, IDs)", df_raw.columns.tolist(),
                                 default=[c for c in label_cols_km if c])
        k_val = st.slider("Number of clusters (K)", 2, 12, 3)
        show_elbow = st.checkbox("Show Elbow & Silhouette plots", value=True)

    X_km, _ = preprocess(df_raw, drop_cols=drop_km)
    
    # PCA for visualization
    pca2 = PCA(n_components=2, random_state=42)
    Xv = pca2.fit_transform(X_km)
    
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_km)
    sil = silhouette_score(X_km, km_labels) if k_val > 1 else 0

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{sil:.3f}</div>
            <div class="lbl">Silhouette Score</div>
        </div>
        <div class="metric-card">
            <div class="val">{km.inertia_:,.0f}</div>
            <div class="lbl">Inertia (WCSS)</div>
        </div>
        """, unsafe_allow_html=True)

        if sil > 0.5:
            st.markdown('<div class="insight-box">✅ Excellent clustering — clusters are well-separated</div>', unsafe_allow_html=True)
        elif sil > 0.25:
            st.markdown('<div class="insight-box warning">⚠️ Moderate clustering — some overlap between clusters</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box warning">⚠️ Weak clustering — try a different K or algorithm</div>', unsafe_allow_html=True)

    with col2:
        plot_df = pd.DataFrame(Xv, columns=["PC1", "PC2"])
        plot_df["Cluster"] = km_labels.astype(str)
        
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                        title=f"K-Means Clusters (K={k_val}) — PCA 2D View",
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        template="plotly_dark")
        
        # Add centroids projected to PCA space
        centroids_2d = pca2.transform(km.cluster_centers_)
        fig.add_trace(go.Scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1],
                                mode='markers', marker=dict(symbol='x', size=14, color='white', line=dict(width=2)),
                                name='Centroids'))
        fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        st.plotly_chart(fig, use_container_width=True)

    if show_elbow:
        st.markdown('<div class="section-header">Elbow Method & Silhouette Scores</div>', unsafe_allow_html=True)
        k_range = list(range(2, 11))
        with st.spinner("Computing elbow curve..."):
            ks, inertias, sils = elbow_data(X_km, k_range)
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig_e = px.line(x=ks, y=inertias, markers=True, title="Elbow Method (Inertia vs K)",
                           labels={"x": "K", "y": "Inertia"}, template="plotly_dark")
            fig_e.add_vline(x=k_val, line_dash="dash", line_color="#4fd1c5")
            fig_e.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
            st.plotly_chart(fig_e, use_container_width=True)
        with col_b:
            fig_s = px.line(x=ks, y=sils, markers=True, title="Silhouette Score vs K",
                           labels={"x": "K", "y": "Silhouette Score"}, template="plotly_dark",
                           color_discrete_sequence=["#4fd1c5"])
            fig_s.add_vline(x=k_val, line_dash="dash", line_color="#f59e0b")
            fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)
    df_with_cluster = df_raw.copy()
    df_with_cluster["Cluster"] = km_labels
    summary = df_with_cluster.groupby("Cluster").mean(numeric_only=True).round(2)
    st.dataframe(summary, use_container_width=True)

    cluster_show = st.selectbox("Browse rows in cluster:", range(k_val))
    st.dataframe(df_raw[km_labels == cluster_show].head(20), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: HIERARCHICAL
# ─────────────────────────────────────────────
elif page == "🌳 Hierarchical Clustering":
    st.markdown('<div class="topic-pill">Hierarchical Clustering</div>', unsafe_allow_html=True)
    st.markdown("## 🌳 Hierarchical Clustering")

    st.markdown("""
    <div class="algo-box">
        <strong>How it works:</strong> Builds a hierarchy of clusters as a tree (dendrogram). 
        <em>Agglomerative</em> (bottom-up): starts with every point as its own cluster and merges the two closest pairs repeatedly.<br><br>
        <strong>Linkage methods:</strong> Ward (minimizes variance), Complete (max distance), Average, Single.<br>
        <strong>Key advantage:</strong> You can cut the dendrogram at any height to get any number of clusters.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        label_cols_h = [c for c in df_raw.columns if df_raw[c].dtype == 'object' or c == label_col]
        drop_h = st.multiselect("Exclude columns", df_raw.columns.tolist(), default=[c for c in label_cols_h if c], key="hc_drop")
        n_clus_h = st.slider("Number of clusters", 2, 10, 3, key="hc_k")
        linkage_m = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])

    X_h, _ = preprocess(df_raw, drop_cols=drop_h)
    
    # Subsample for dendrogram if too large
    n_dendro = min(200, len(X_h))
    X_dendro = X_h[:n_dendro]
    
    agg = AgglomerativeClustering(n_clusters=n_clus_h, linkage=linkage_m)
    h_labels = agg.fit_predict(X_h)
    sil_h = silhouette_score(X_h, h_labels) if n_clus_h > 1 else 0

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{sil_h:.3f}</div>
            <div class="lbl">Silhouette Score</div>
        </div>
        """, unsafe_allow_html=True)
        sizes = pd.Series(h_labels).value_counts().sort_index()
        for c, sz in sizes.items():
            st.markdown(f'<div class="metric-card"><div class="val">{sz}</div><div class="lbl">Cluster {c} size</div></div>', unsafe_allow_html=True)

    with col2:
        pca2h = PCA(n_components=2, random_state=42)
        Xvh = pca2h.fit_transform(X_h)
        plot_df_h = pd.DataFrame(Xvh, columns=["PC1", "PC2"])
        plot_df_h["Cluster"] = h_labels.astype(str)
        fig_h = px.scatter(plot_df_h, x="PC1", y="PC2", color="Cluster",
                          title=f"Hierarchical Clusters (n={n_clus_h}, linkage={linkage_m})",
                          color_discrete_sequence=px.colors.qualitative.Pastel, template="plotly_dark")
        fig_h.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<div class="section-header">Dendrogram (first 200 rows)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
        The <strong>dendrogram</strong> shows the merge history. The Y-axis is the distance at which clusters were merged.
        A horizontal cut at any height gives you that many clusters.
    </div>
    """, unsafe_allow_html=True)
    
    fig_d, ax = plt.subplots(figsize=(14, 4))
    fig_d.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    Z = linkage(X_dendro, method=linkage_m)
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=45, color_threshold=0.7 * max(Z[:, 2]))
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#2a3a50')
    ax.spines['left'].set_color('#2a3a50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Hierarchical Clustering Dendrogram", color='white', fontsize=13)
    ax.set_xlabel("Sample Index / Cluster", color='#7a8fa6')
    ax.set_ylabel("Distance", color='#7a8fa6')
    st.pyplot(fig_d)

# ─────────────────────────────────────────────
# PAGE: DBSCAN
# ─────────────────────────────────────────────
elif page == "🔵 DBSCAN":
    st.markdown('<div class="topic-pill">Density-Based Clustering</div>', unsafe_allow_html=True)
    st.markdown("## 🔵 DBSCAN")

    st.markdown("""
    <div class="algo-box">
        <strong>How it works:</strong> DBSCAN groups points that are closely packed together and marks points in 
        low-density regions as <em>outliers/noise</em> (labeled -1).<br><br>
        <strong>Key parameters:</strong><br>
        &bull; <strong>eps</strong> — neighborhood radius (smaller = tighter clusters)<br>
        &bull; <strong>min_samples</strong> — minimum points to form a core point<br><br>
        <strong>Key advantage:</strong> Finds clusters of <em>any shape</em> and detects outliers automatically.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        label_cols_d = [c for c in df_raw.columns if df_raw[c].dtype == 'object']
        drop_d = st.multiselect("Exclude columns", df_raw.columns.tolist(), default=label_cols_d, key="db_drop")
        eps_val = st.slider("eps (neighborhood radius)", 0.1, 5.0, 0.8, 0.1)
        min_s = st.slider("min_samples", 2, 20, 5)

    X_db, _ = preprocess(df_raw, drop_cols=drop_d)
    db = DBSCAN(eps=eps_val, min_samples=min_s)
    db_labels = db.fit_predict(X_db)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)

    with col1:
        st.markdown(f"""
        <div class="metric-card"><div class="val">{n_clusters_db}</div><div class="lbl">Clusters Found</div></div>
        <div class="metric-card"><div class="val">{n_noise}</div><div class="lbl">Outliers Detected</div></div>
        <div class="metric-card"><div class="val">{n_noise/len(db_labels)*100:.1f}%</div><div class="lbl">Noise Ratio</div></div>
        """, unsafe_allow_html=True)
        
        if n_clusters_db == 0:
            st.markdown('<div class="insight-box warning">⚠️ All points are noise! Increase eps or decrease min_samples.</div>', unsafe_allow_html=True)
        elif n_clusters_db == 1:
            st.markdown('<div class="insight-box warning">⚠️ Only 1 cluster found. Try reducing eps.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-box">✅ Found {n_clusters_db} clusters with {n_noise} outliers.</div>', unsafe_allow_html=True)

    with col2:
        pca2d = PCA(n_components=2, random_state=42)
        Xvd = pca2d.fit_transform(X_db)
        plot_df_d = pd.DataFrame(Xvd, columns=["PC1", "PC2"])
        plot_df_d["Cluster"] = [f"Noise" if l == -1 else f"Cluster {l}" for l in db_labels]
        
        color_map = {"Noise": "#ff4444"}
        colors = px.colors.qualitative.Vivid
        for i in range(n_clusters_db):
            color_map[f"Cluster {i}"] = colors[i % len(colors)]
        
        fig_db = px.scatter(plot_df_d, x="PC1", y="PC2", color="Cluster",
                           title=f"DBSCAN (eps={eps_val}, min_samples={min_s})",
                           color_discrete_map=color_map, template="plotly_dark")
        fig_db.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        st.plotly_chart(fig_db, use_container_width=True)

    st.markdown('<div class="section-header">DBSCAN vs K-Means — When to Use Each</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Feature": ["Shape of clusters", "Outlier detection", "Need to specify K", "Scalability", "Best for"],
        "K-Means": ["Spherical only", "❌ No", "✅ Yes (required)", "⚡ Very fast", "Clean, round clusters"],
        "DBSCAN": ["Any shape", "✅ Yes", "❌ No", "🐢 Slower", "Irregular shapes, noise in data"]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE: PCA
# ─────────────────────────────────────────────
elif page == "📉 PCA — Dimensionality Reduction":
    st.markdown('<div class="topic-pill">Dimensionality Reduction</div>', unsafe_allow_html=True)
    st.markdown("## 📉 Principal Component Analysis (PCA)")

    st.markdown("""
    <div class="algo-box">
        <strong>What it does:</strong> PCA finds the directions (principal components) of maximum variance in your data 
        and projects data onto fewer dimensions — losing as little information as possible.<br><br>
        <strong>Why it's useful:</strong> High-dimensional data is impossible to visualize. PCA lets you compress 
        50 features into 2–3 components while keeping the most important patterns.
    </div>
    """, unsafe_allow_html=True)

    label_cols_p = [c for c in df_raw.columns if df_raw[c].dtype == 'object']
    drop_p = st.multiselect("Exclude columns", df_raw.columns.tolist(), default=label_cols_p, key="pca_drop")
    
    X_pca, feat_names = preprocess(df_raw, drop_cols=drop_p)
    max_comp = min(X_pca.shape[1], X_pca.shape[0], 15)
    n_comp = st.slider("Number of principal components", 2, max_comp, min(max_comp, 5))

    pca_full = PCA(n_components=n_comp, random_state=42)
    Xp_full = pca_full.fit_transform(X_pca)
    
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    col1, col2 = st.columns(2)
    with col1:
        fig_ev = go.Figure()
        fig_ev.add_bar(x=[f"PC{i+1}" for i in range(n_comp)], y=evr * 100,
                      marker_color='#4fd1c5', name="Individual")
        fig_ev.add_scatter(x=[f"PC{i+1}" for i in range(n_comp)], y=cumvar * 100,
                          mode='lines+markers', line=dict(color='#f59e0b', width=2),
                          name="Cumulative", yaxis='y')
        fig_ev.update_layout(title="Explained Variance per Component",
                            yaxis_title="Variance Explained (%)", template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        st.plotly_chart(fig_ev, use_container_width=True)

        comp_for_95 = np.argmax(cumvar >= 0.95) + 1 if any(cumvar >= 0.95) else n_comp
        st.markdown(f'<div class="insight-box">ℹ️ <strong>{comp_for_95} components</strong> explain 95%+ of variance (from {X_pca.shape[1]} original features)</div>', unsafe_allow_html=True)

    with col2:
        # Biplot — loadings heatmap
        loadings = pd.DataFrame(pca_full.components_[:min(n_comp, 5)].T,
                                index=feat_names,
                                columns=[f"PC{i+1}" for i in range(min(n_comp, 5))])
        fig_load = px.imshow(loadings.T, title="Feature Loadings (which features drive each PC)",
                            color_continuous_scale="RdBu_r", template="plotly_dark",
                            aspect="auto")
        fig_load.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_load, use_container_width=True)

    # 2D scatter with optional class coloring
    st.markdown('<div class="section-header">PCA 2D Scatter</div>', unsafe_allow_html=True)
    color_by = st.selectbox("Color points by (optional label)", [None] + [c for c in df_raw.columns if df_raw[c].dtype == 'object'])
    
    scatter_df = pd.DataFrame(Xp_full[:, :2], columns=["PC1", "PC2"])
    if color_by and color_by in df_raw.columns:
        scatter_df["Label"] = df_raw[color_by].astype(str).values
        fig_sc = px.scatter(scatter_df, x="PC1", y="PC2", color="Label",
                           title="PCA 2D Projection", template="plotly_dark",
                           color_discrete_sequence=px.colors.qualitative.Vivid)
    else:
        fig_sc = px.scatter(scatter_df, x="PC1", y="PC2",
                           title="PCA 2D Projection", template="plotly_dark")
    
    fig_sc.update_layout(height=430, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: t-SNE
# ─────────────────────────────────────────────
elif page == "🌀 t-SNE Visualization":
    st.markdown('<div class="topic-pill">Non-Linear Dimensionality Reduction</div>', unsafe_allow_html=True)
    st.markdown("## 🌀 t-SNE Visualization")

    st.markdown("""
    <div class="algo-box">
        <strong>What it does:</strong> t-SNE maps high-dimensional data to 2D by preserving local neighborhood structure.
        Similar points in high dimensions end up close together in 2D.<br><br>
        <strong>vs PCA:</strong> PCA is linear and preserves global variance. t-SNE is non-linear and better reveals 
        cluster structures visually — but the axes have no direct meaning.<br><br>
        <strong>⚠️ Note:</strong> t-SNE is slow on large datasets. Capped at 2,000 rows here.
    </div>
    """, unsafe_allow_html=True)

    label_cols_t = [c for c in df_raw.columns if df_raw[c].dtype == 'object']
    drop_t = st.multiselect("Exclude columns", df_raw.columns.tolist(), default=label_cols_t, key="tsne_drop")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        perplexity = st.slider("Perplexity (5–50)", 5, 50, 30)
        color_tsne = st.selectbox("Color by", [None] + [c for c in df_raw.columns if df_raw[c].dtype == 'object'], key="tsne_color")
        run_tsne = st.button("▶ Run t-SNE", type="primary")

    if run_tsne:
        X_tsne, _ = preprocess(df_raw, drop_cols=drop_t)
        max_rows = min(2000, len(X_tsne))
        X_sub = X_tsne[:max_rows]
        
        with st.spinner(f"Running t-SNE on {max_rows} rows... (may take ~30 seconds)"):
            pca_pre = PCA(n_components=min(50, X_sub.shape[1]), random_state=42)
            X_pre = pca_pre.fit_transform(X_sub)
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
            except TypeError:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
            X_2d = tsne.fit_transform(X_pre)
        
        tsne_df = pd.DataFrame(X_2d, columns=["Dim1", "Dim2"])
        if color_tsne and color_tsne in df_raw.columns:
            tsne_df["Label"] = df_raw[color_tsne].iloc[:max_rows].astype(str).values
            fig_t = px.scatter(tsne_df, x="Dim1", y="Dim2", color="Label",
                              title=f"t-SNE (perplexity={perplexity}, n={max_rows})",
                              template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
        else:
            fig_t = px.scatter(tsne_df, x="Dim1", y="Dim2",
                              title=f"t-SNE (perplexity={perplexity}, n={max_rows})", template="plotly_dark")
        
        fig_t.update_traces(marker=dict(size=4, opacity=0.7))
        fig_t.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        with col2:
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        with col2:
            st.info("👆 Click **Run t-SNE** to generate the visualization. t-SNE is computationally intensive so it doesn't run automatically.")

# ─────────────────────────────────────────────
# PAGE: MOVIE RECOMMENDER
# ─────────────────────────────────────────────
elif page == "🎬 Movie Recommender":
    st.markdown('<div class="topic-pill">Clustering-Based Recommendation</div>', unsafe_allow_html=True)
    st.markdown("## 🎬 Movie Recommender")

    st.markdown("""
    <div class="algo-box">
        <strong>How it works:</strong> Movies are clustered based on their features (genre, rating, year, etc.) using K-Means.
        When you select a movie, the app finds other movies in the <em>same cluster</em> — movies with similar characteristics.<br><br>
        This is <strong>content-based filtering</strong> powered by unsupervised clustering.
    </div>
    """, unsafe_allow_html=True)

    movies = get_movies()
    if movies is None:
        st.error("Sample_Movies_Dataset.csv not found. Please ensure the file is in the app directory.")
        st.stop()

    st.dataframe(movies.head(), use_container_width=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        n_movie_clusters = st.slider("Number of movie clusters", 2, 10, 4)
        feature_cols = st.multiselect("Features to cluster on",
                                      [c for c in movies.columns if c != 'title'],
                                      default=[c for c in ['genre', 'year', 'rating', 'duration'] if c in movies.columns])
    
    if not feature_cols:
        st.warning("Please select at least one feature.")
        st.stop()

    X_movies, _ = preprocess(movies, drop_cols=[c for c in movies.columns if c not in feature_cols])
    km_movies = KMeans(n_clusters=n_movie_clusters, random_state=42, n_init=10)
    movie_labels = km_movies.fit_predict(X_movies)
    movies_clustered = movies.copy()
    movies_clustered["Cluster"] = movie_labels

    with col2:
        pca_mv = PCA(n_components=2, random_state=42)
        Xv_mv = pca_mv.fit_transform(X_movies)
        mv_df = pd.DataFrame(Xv_mv, columns=["PC1", "PC2"])
        mv_df["Cluster"] = movie_labels.astype(str)
        mv_df["Title"] = movies["title"].values if "title" in movies.columns else range(len(movies))
        fig_mv = px.scatter(mv_df, x="PC1", y="PC2", color="Cluster", hover_name="Title",
                           title="Movies Clustered by Features",
                           color_discrete_sequence=px.colors.qualitative.Bold, template="plotly_dark")
        fig_mv.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        st.plotly_chart(fig_mv, use_container_width=True)

    st.markdown('<div class="section-header">🔍 Get Movie Recommendations</div>', unsafe_allow_html=True)
    if "title" in movies.columns:
        selected = st.selectbox("Select a movie", movies["title"].tolist())
        selected_idx = movies[movies["title"] == selected].index[0]
        cluster_id = movie_labels[selected_idx]
        similar = movies_clustered[movies_clustered["Cluster"] == cluster_id]
        similar = similar[similar.index != selected_idx]
        
        st.success(f"**{selected}** is in Cluster {cluster_id}. Here are {len(similar)} similar movies:")
        st.dataframe(similar.drop(columns=["Cluster"]), use_container_width=True)
    else:
        st.info("No 'title' column found. Please include a title column for recommendations.")

# ─────────────────────────────────────────────
# PAGE: CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────
elif page == "👤 Customer Segmentation":
    st.markdown('<div class="topic-pill">Real-World Application</div>', unsafe_allow_html=True)
    st.markdown("## 👤 Customer Segmentation")

    st.markdown("""
    <div class="algo-box">
        <strong>Business use case:</strong> Group customers by Age, Income, Spending Score, and other features.
        Each cluster represents a customer segment that can be targeted with different marketing strategies.<br><br>
        This uses your <strong>mixed_dataset.csv</strong> with all three algorithms for comparison.
    </div>
    """, unsafe_allow_html=True)

    mixed = get_mixed()
    if mixed is None:
        st.warning("mixed_dataset.csv not found.")
        st.stop()

    numeric_features = mixed.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = mixed.select_dtypes(include='object').columns.tolist()

    col1, col2 = st.columns([1, 2])
    with col1:
        sel_features = st.multiselect("Select features for segmentation", numeric_features,
                                      default=numeric_features[:4])
        n_seg = st.slider("Number of segments", 2, 8, 4)
        algo = st.selectbox("Algorithm", ["K-Means", "Hierarchical (Ward)", "DBSCAN"])

    if not sel_features:
        st.warning("Select at least 2 features.")
        st.stop()

    X_seg = StandardScaler().fit_transform(mixed[sel_features].fillna(mixed[sel_features].median()))

    if algo == "K-Means":
        model_seg = KMeans(n_clusters=n_seg, random_state=42, n_init=10)
        seg_labels = model_seg.fit_predict(X_seg)
    elif algo == "Hierarchical (Ward)":
        model_seg = AgglomerativeClustering(n_clusters=n_seg, linkage='ward')
        seg_labels = model_seg.fit_predict(X_seg)
    else:
        eps_seg = st.slider("DBSCAN eps", 0.3, 3.0, 0.8, 0.1, key="seg_eps")
        model_seg = DBSCAN(eps=eps_seg, min_samples=10)
        seg_labels = model_seg.fit_predict(X_seg)

    with col2:
        pca_seg = PCA(n_components=2, random_state=42)
        Xv_seg = pca_seg.fit_transform(X_seg)
        seg_df = pd.DataFrame(Xv_seg, columns=["PC1", "PC2"])
        seg_df["Segment"] = [f"Segment {l}" if l != -1 else "Outlier" for l in seg_labels]
        fig_seg = px.scatter(seg_df, x="PC1", y="PC2", color="Segment",
                            title=f"Customer Segments — {algo}",
                            color_discrete_sequence=px.colors.qualitative.Safe, template="plotly_dark")
        fig_seg.update_layout(height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,1)')
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown('<div class="section-header">Segment Profiles</div>', unsafe_allow_html=True)
    seg_result = mixed[sel_features].copy()
    seg_result["Segment"] = seg_labels
    seg_summary = seg_result[seg_result["Segment"] != -1].groupby("Segment").agg(['mean', 'count']).round(2)
    st.dataframe(seg_summary, use_container_width=True)

    # Radar chart for segment comparison
    if len(sel_features) >= 3:
        st.markdown('<div class="section-header">Segment Radar Chart</div>', unsafe_allow_html=True)
        seg_means = seg_result[seg_result["Segment"] != -1].groupby("Segment")[sel_features].mean()
        # Normalize for radar
        seg_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-8)
        
        fig_radar = go.Figure()
        colors_r = px.colors.qualitative.Safe
        for i, (seg_id, row) in enumerate(seg_norm.iterrows()):
            vals = row.tolist() + [row.tolist()[0]]
            cats = sel_features + [sel_features[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself', name=f"Segment {seg_id}",
                line=dict(color=colors_r[i % len(colors_r)])
            ))
        fig_radar.update_layout(polar=dict(bgcolor='rgba(15,20,35,1)',
                                           radialaxis=dict(visible=True, gridcolor='#2a3a50'),
                                           angularaxis=dict(gridcolor='#2a3a50')),
                               paper_bgcolor='rgba(0,0,0,0)', template="plotly_dark",
                               title="Segment Profiles (normalized)", height=450)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-header">Business Interpretation</div>', unsafe_allow_html=True)
    for seg_id in sorted(set(seg_labels)):
        if seg_id == -1:
            continue
        seg_data = mixed[sel_features][seg_labels == seg_id].mean().round(1)
        st.markdown(f"""
        <div class="algo-box">
            <strong>Segment {seg_id}</strong> ({(seg_labels == seg_id).sum()} customers)<br>
            {" | ".join([f"{k}: <strong>{v}</strong>" for k,v in seg_data.items()])}
        </div>
        """, unsafe_allow_html=True)
