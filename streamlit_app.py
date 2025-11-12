# streamlit_app.py — Robust Unsupervised Recommender (final, fixed)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Try plotly; if missing we'll show a warning but continue
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("mixed_dataset.csv")

st.set_page_config(page_title="🎬 Unsupervised Recommender", layout="wide")
st.title("🎬 Unsupervised Movie/Product Recommender — KMeans + PCA")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data
def load_data(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_resource
def train_model(df: pd.DataFrame, n_clusters: int = 5):
    # choose numeric columns and drop rows with missing numeric values
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if numeric_df.empty:
        return None

    numeric_index = numeric_df.index.to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_pca)

    return {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "labels": np.asarray(labels),
        "X_pca": np.asarray(X_pca),
        "numeric_index": np.asarray(numeric_index),
        "numeric_columns": numeric_df.columns.tolist()
    }

def load_model(path: Path):
    if not path.exists():
        return None
    try:
        loaded = joblib.load(path)
    except Exception:
        return None
    if loaded is None or not isinstance(loaded, dict):
        return None
    return loaded

def repair_old_model(bundle):
    """
    Normalize a few common legacy key names into the expected bundle keys.
    Returns (bundle, repaired_flag).
    """
    if bundle is None or not isinstance(bundle, dict):
        return None, False

    repaired = False

    if "labels" not in bundle:
        for alt in ("label", "y_labels", "clusters"):
            if alt in bundle:
                bundle["labels"] = bundle.pop(alt)
                repaired = True
                break

    if "X_pca" not in bundle:
        for alt in ("x_pca", "xpca", "X_PCA", "Xpca"):
            if alt in bundle:
                bundle["X_pca"] = bundle.pop(alt)
                repaired = True
                break

    if "numeric_index" not in bundle:
        if "index" in bundle:
            bundle["numeric_index"] = np.asarray(bundle.pop("index"))
            repaired = True
        elif "orig" in bundle and hasattr(bundle["orig"], "index"):
            bundle["numeric_index"] = np.asarray(bundle.pop("orig").index)
            repaired = True

    if "numeric_columns" not in bundle:
        if "data" in bundle and hasattr(bundle["data"], "columns"):
            bundle["numeric_columns"] = bundle["data"].columns.tolist()
            repaired = True

    return bundle, repaired

# ----------------------------
# Load dataset (upload or local)
# ----------------------------
uploaded = st.file_uploader("📤 Upload CSV (optional). If not uploaded, it will use mixed_dataset.csv", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded CSV loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    df = load_data(DATA_PATH)
    if df is None:
        st.error("❌ No CSV found. Please upload a dataset to continue.")
        st.stop()
    else:
        st.info(f"ℹ️ Loaded dataset from {DATA_PATH}")

st.write("### 📊 Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Load saved model (if any) and repair
# ----------------------------
model_bundle = load_model(MODEL_PATH)
if model_bundle is not None:
    model_bundle, was_repaired = repair_old_model(model_bundle)
    if was_repaired:
        try:
            joblib.dump(model_bundle, MODEL_PATH)
            st.info("Repaired old model.pkl structure and saved updated model.")
        except Exception:
            st.warning("Repaired model in memory but failed to overwrite model.pkl on disk.")

# ----------------------------
# If no model, show training UI
# ----------------------------
if model_bundle is None:
    st.info("No existing saved model found. You can train a new KMeans model below.")
    n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            new_bundle = train_model(df, n_clusters)
            if new_bundle:
                model_bundle = new_bundle
                try:
                    joblib.dump(model_bundle, MODEL_PATH)
                    st.success("🎉 Model trained and saved successfully!")
                except Exception:
                    st.warning("Model trained in memory but failed to save model.pkl to disk.")
            else:
                st.error("Failed to train model. Dataset may not have numeric columns or has many missing values.")

# If still no model, stop
if model_bundle is None:
    st.warning("Train or load a model to continue.")
    st.stop()

# ----------------------------
# SAFETY: must be dict before using 'in' or .get()
# ----------------------------
if not isinstance(model_bundle, dict):
    st.error("Loaded model is not in the expected format (not a dict). Delete model.pkl and retrain.")
    st.stop()

required_keys = ("labels", "X_pca", "numeric_index")
missing = [k for k in required_keys if k not in model_bundle or model_bundle.get(k) is None]
if missing:
    st.error(f"Loaded model is missing required components: {missing}. Delete model.pkl and retrain.")
    st.stop()

# Convert to numpy arrays and sanity-check shapes
labels = np.asarray(model_bundle.get("labels"))
X_pca = np.asarray(model_bundle.get("X_pca"))
numeric_index = np.asarray(model_bundle.get("numeric_index"))

if labels.ndim != 1 or X_pca.ndim < 1 or numeric_index.ndim != 1:
    st.error("Loaded model components have unexpected shapes. Retrain the model.")
    st.stop()

# ----------------------------
# PCA visualization
# ----------------------------
st.subheader("📈 PCA Visualization of Clusters")
try:
    if X_pca.shape[1] == 1:
        pc1 = X_pca[:, 0]
        pc2 = np.zeros_like(pc1)
        plot_df = pd.DataFrame({"PC1": pc1, "PC2": pc2})
    else:
        plot_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
except Exception as e:
    st.error(f"Error preparing PCA plot: {e}")
    st.stop()

plot_df["cluster"] = labels.astype(str)

if PLOTLY_AVAILABLE:
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster",
                     title="PCA Clusters Visualization", width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Plotly not available — PCA scatter cannot be displayed. Install plotly to enable visualization.")

# Cluster sizes
st.write("### 📦 Cluster Sizes")
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.write(cluster_counts)

# Examples from cluster mapped to original df
st.write("### 🔎 Show examples from a selected cluster (original rows)")
cluster_choice = st.selectbox("Choose cluster", sorted(plot_df["cluster"].unique().tolist()))
try:
    cluster_id_int = int(cluster_choice)
except Exception:
    st.error("Unexpected cluster selection.")
    st.stop()

cluster_positions = np.where(labels == cluster_id_int)[0]
original_indices = numeric_index[cluster_positions]
if len(original_indices) > 0:
    try:
        st.dataframe(df.loc[original_indices].head(20))
    except Exception as e:
        st.error(f"Error selecting original rows: {e}")
else:
    st.info("No examples found for this cluster in the original dataset.")

# ----------------------------
# Recommendation engine
# ----------------------------
st.subheader("🎬 Recommendation Engine")
columns = df.columns.tolist()
id_col = st.selectbox("Select an identifier column (e.g., product name, movie title, ID):", [None] + columns)

if id_col:
    options = df[id_col].dropna().unique().tolist()
    if not options:
        st.error("No non-empty values found in selected identifier column.")
    else:
        selected_item = st.selectbox("Select an item to get recommendations:", options)
        selected_indices_all = df.index[df[id_col] == selected_item].tolist()
        if len(selected_indices_all) == 0:
            st.error("Selected item not found in the dataset (unexpected).")
        else:
            selected_index = selected_indices_all[0]
            pos = np.where(numeric_index == selected_index)[0]
            if pos.size == 0:
                st.warning("The selected item was not used for clustering because numeric features were missing or dropped.")
            else:
                pos = int(pos[0])
                cluster_id = labels[pos]
                same_cluster_positions = np.where(labels == cluster_id)[0]
                same_cluster_original_indices = numeric_index[same_cluster_positions]
                same_cluster_original_indices = same_cluster_original_indices[same_cluster_original_indices != selected_index]
                similar_items_df = df.loc[same_cluster_original_indices]
                st.success(f"✅ Found {len(similar_items_df)} similar items in Cluster {cluster_id}")
                if len(similar_items_df) == 0:
                    st.info("No other items in this cluster to recommend.")
                else:
                    st.dataframe(similar_items_df.head(20))
else:
    st.info("👆 Choose an identifier column to enable recommendations.")

st.write("---")
st.caption("Built with ❤️ using Streamlit, KMeans, and PCA")
