# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# plotting: try plotly first, else use matplotlib
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("mixed_dataset.csv")  # adjust if your CSV is in a different folder

st.set_page_config(page_title="Unsupervised Recommender", layout="wide")
st.title("Unsupervised Recommender — demo (safe for missing Plotly)")

@st.cache_data
def load_data(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_saved_model(path: Path):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load saved model: {e}")
            return None
    return None

def train_model(df, n_clusters=5):
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0)
    if numeric.shape[0] == 0 or numeric.shape[1] == 0:
        st.error("No numeric columns available for training. Please provide a CSV with numeric features.")
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)
    n_components = min(10, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(Xp)

    bundle = {"model": model, "scaler": scaler, "pca": pca, "labels": labels, "Xp": Xp, "orig": numeric}
    return bundle

# --- Load dataset (upload or local) ---
uploaded = st.file_uploader("Upload CSV (optional). If omitted, app will try to load mixed_dataset.csv", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    df = load_data(DATA_PATH)
    if df is None:
        st.warning("No dataset found at 'mixed_dataset.csv'. Upload a CSV to continue.")
        st.stop()
    else:
        st.info(f"Loaded dataset from {DATA_PATH}")

st.write("### Dataset sample")
st.dataframe(df.head())

# --- Load or train model ---
model_bundle = load_saved_model(MODEL_PATH)
if model_bundle:
    st.success("Loaded saved model from model.pkl")
else:
    st.info("No saved model found. You can train a new KMeans model below.")
    n_clusters = st.slider("Number of clusters", 2, 12, 5)
    if st.button("Train KMeans"):
        with st.spinner("Training..."):
            trained = train_model(df, n_clusters=n_clusters)
            if trained:
                joblib.dump(trained, MODEL_PATH)
                st.success("Model trained and saved to model.pkl")
                model_bundle = trained

# --- Visualize results ---
if model_bundle:
    labels = model_bundle["labels"]
    Xp = model_bundle["Xp"]
    orig = model_bundle["orig"]

    plot_df = pd.DataFrame(Xp[:, :2], columns=["PC1", "PC2"])
    plot_df["cluster"] = labels.astype(str)

    st.write("### PCA visualization of clusters")
    if PLOTLY_AVAILABLE:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", title="PCA: clusters")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        groups = plot_df.groupby("cluster")
        for name, group in groups:
            ax.scatter(group["PC1"], group["PC2"], label=str(name), alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA: clusters (matplotlib fallback)")
        ax.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

    st.write("### Cluster sizes")
    st.write(pd.Series(labels).value_counts().sort_index())

    st.write("### Show examples from a selected cluster")
    cluster_choice = st.selectbox("Choose cluster", sorted(plot_df["cluster"].unique().tolist()))
    idxs = plot_df[plot_df["cluster"] == cluster_choice].index
    st.dataframe(orig.iloc[idxs].head(20))
