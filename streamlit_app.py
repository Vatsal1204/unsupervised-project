# streamlit_app.py — Unsupervised Recommender (auto-encode + auto-train)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Plotly optional
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("mixed_dataset.csv")

st.set_page_config(page_title="🎬 Unsupervised Recommender", layout="wide")
st.title("🎬 Unsupervised Movie/Product Recommender — KMeans + PCA")

# ------------------------
# helpers
# ------------------------
@st.cache_data
def load_data(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def preprocess_for_clustering(df: pd.DataFrame):
    """
    Convert categorical columns to numeric via one-hot encoding,
    keep numeric columns, drop columns with all nulls, drop rows with all-NaN.
    Returns processed_df, original_index_map (array of original df indices used).
    """
    if df is None or df.shape[0] == 0:
        return None, None

    # Keep a copy of original indexes
    orig_idx = df.index.to_numpy()

    # Simple strategy: one-hot encode object (categorical) columns, keep numeric as-is
    try:
        # Limit number of dummy columns by drop_first to reduce dimension
        object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # If there are categorical cols, create dummies; else keep numeric only
        if len(object_cols) > 0:
            # create dummies; drop_first to avoid redundancy
            dummies = pd.get_dummies(df[object_cols], drop_first=True)
            processed = pd.concat([df[numeric_cols], dummies], axis=1)
        else:
            processed = df[numeric_cols].copy()

        # drop columns with all zeros/NaN
        processed = processed.loc[:, (processed.notna().any(axis=0))]

        # drop rows with any NaNs (for simplicity) — alternatively impute
        processed = processed.dropna(axis=0, how="any")
        if processed.empty:
            return None, None

        # Update original indices to those rows that survived
        used_orig_idx = orig_idx[processed.index.to_numpy()]

        return processed, used_orig_idx
    except Exception:
        return None, None

@st.cache_resource
def train_model_from_df(df: pd.DataFrame, n_clusters: int = 5):
    """
    Preprocess df (auto one-hot), scale, PCA, KMeans.
    Returns model bundle or None on failure.
    """
    processed, used_idx = preprocess_for_clustering(df)
    if processed is None or processed.shape[0] == 0:
        return None

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(processed.values)

    # PCA components default: min(5, features)
    n_components = min(5, X_scaled.shape[1])
    if n_components < 1:
        return None
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_pca)

    bundle = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "labels": np.asarray(labels),
        "X_pca": np.asarray(X_pca),
        "numeric_index": np.asarray(used_idx),
        "numeric_columns": processed.columns.tolist()
    }
    return bundle

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
    if bundle is None or not isinstance(bundle, dict):
        return None, False
    repaired = False
    if "labels" not in bundle:
        for alt in ("label", "y_labels", "clusters"):
            if alt in bundle:
                bundle["labels"] = bundle.pop(alt)
                repaired = True; break
    if "X_pca" not in bundle:
        for alt in ("x_pca", "xpca", "X_PCA", "Xpca"):
            if alt in bundle:
                bundle["X_pca"] = bundle.pop(alt)
                repaired = True; break
    if "numeric_index" not in bundle:
        if "index" in bundle:
            bundle["numeric_index"] = np.asarray(bundle.pop("index")); repaired = True
        elif "orig" in bundle and hasattr(bundle["orig"], "index"):
            bundle["numeric_index"] = np.asarray(bundle.pop("orig").index); repaired = True
    if "numeric_columns" not in bundle:
        if "data" in bundle and hasattr(bundle["data"], "columns"):
            bundle["numeric_columns"] = bundle["data"].columns.tolist(); repaired = True
    return bundle, repaired

# ------------------------
# load dataset (upload or local)
# ------------------------
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

# ------------------------
# model load & auto-train option
# ------------------------
model_bundle = load_model(MODEL_PATH)
if model_bundle is not None:
    model_bundle, was_repaired = repair_old_model(model_bundle)
    if was_repaired:
        try:
            joblib.dump(model_bundle, MODEL_PATH)
            st.info("Repaired old model.pkl structure and saved updated model.")
        except Exception:
            st.warning("Repaired model in memory but failed to overwrite model.pkl on disk.")

st.write("---")
st.write("Model options")
auto_train = st.checkbox("Automatically train a fresh model now if model.pkl is missing or incompatible", value=True)

trained_now = False
if (model_bundle is None) and auto_train:
    # Train automatically using uploaded data
    st.info("Auto-training a new model on this dataset (this may take a few seconds)...")
    with st.spinner("Training..."):
        try:
            new_bundle = train_model_from_df(df, n_clusters=5)
            if new_bundle:
                model_bundle = new_bundle
                joblib.dump(model_bundle, MODEL_PATH)
                st.success("Auto-trained and saved new model to model.pkl")
                trained_now = True
            else:
                st.error("Auto-train failed: dataset may not have usable columns after preprocessing.")
        except Exception as e:
            st.error(f"Auto-train error: {e}")

# Also allow manual training with UI
if model_bundle is None:
    st.info("No saved model available. You can manually train below.")
    n_clusters = st.slider("Select number of clusters:", 2, 12, 5)
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            new_bundle = train_model_from_df(df, n_clusters=n_clusters)
            if new_bundle:
                model_bundle = new_bundle
                try:
                    joblib.dump(model_bundle, MODEL_PATH)
                    st.success("Model trained and saved to model.pkl")
                except Exception:
                    st.warning("Model trained in memory but failed to save to disk.")
            else:
                st.error("Training failed. Check dataset for sufficient data after preprocessing.")

if model_bundle is None:
    st.warning("No model available. Upload a different CSV or enable training.")
    st.stop()

# ------------------------
# safety checks
# ------------------------
if not isinstance(model_bundle, dict):
    st.error("Loaded model is not a dict. Delete model.pkl and retrain.")
    st.stop()

required_keys = ("labels", "X_pca", "numeric_index")
missing = [k for k in required_keys if k not in model_bundle or model_bundle.get(k) is None]
if missing:
    # If an old bundle has alternate keys, attempt quick repair here
    # (we already tried repair_old_model earlier)
    st.error(f"Loaded model is missing required components: {missing}. Delete model.pkl and retrain (or enable auto-train).")
    st.stop()

labels = np.asarray(model_bundle.get("labels"))
X_pca = np.asarray(model_bundle.get("X_pca"))
numeric_index = np.asarray(model_bundle.get("numeric_index"))

if labels.ndim != 1 or X_pca.ndim < 1 or numeric_index.ndim != 1:
    st.error("Loaded model components have unexpected shapes. Retrain model.")
    st.stop()

# ------------------------
# PCA visualization
# ------------------------
st.subheader("📈 PCA Visualization of Clusters")
try:
    if X_pca.shape[1] == 1:
        pc1 = X_pca[:, 0]; pc2 = np.zeros_like(pc1)
        plot_df = pd.DataFrame({"PC1": pc1, "PC2": pc2})
    else:
        plot_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
except Exception as e:
    st.error(f"Error preparing PCA plot: {e}")
    st.stop()

plot_df["cluster"] = labels.astype(str)

if PLOTLY_AVAILABLE:
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", title="PCA Clusters Visualization")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write(plot_df.head())  # fallback simple view

# cluster sizes
st.write("### 📦 Cluster Sizes")
st.write(pd.Series(labels).value_counts().sort_index())

# show some examples mapped to original indices
st.write("### 🔎 Examples from a selected cluster (original rows)")
cluster_choice = st.selectbox("Choose cluster", sorted(plot_df["cluster"].unique().tolist()))
try:
    cluster_id_int = int(cluster_choice)
except Exception:
    st.error("Invalid cluster choice.")
    st.stop()

cluster_positions = np.where(labels == cluster_id_int)[0]
orig_indices = numeric_index[cluster_positions]
if len(orig_indices) > 0:
    try:
        st.dataframe(df.loc[orig_indices].head(20))
    except Exception as e:
        st.error(f"Error showing original rows: {e}")
else:
    st.info("No examples found for this cluster in the original dataset.")

# ------------------------
# recommendation engine
# ------------------------
st.subheader("🎬 Recommendation Engine")
columns = df.columns.tolist()
id_col = st.selectbox("Select an identifier column (e.g., product name, movie title, ID):", [None] + columns)

if id_col:
    options = df[id_col].dropna().unique().tolist()
    if not options:
        st.error("No non-empty values found in selected identifier column.")
    else:
        selected_item = st.selectbox("Select an item to get recommendations:", options)
        indices = df.index[df[id_col] == selected_item].tolist()
        if len(indices) == 0:
            st.error("Selected item not found (unexpected).")
        else:
            sel_idx = indices[0]
            pos = np.where(numeric_index == sel_idx)[0]
            if pos.size == 0:
                st.warning("Selected item wasn't used for clustering (missing numeric/encoded features).")
            else:
                pos = int(pos[0])
                cluster_id = labels[pos]
                same_positions = np.where(labels == cluster_id)[0]
                same_orig_indices = numeric_index[same_positions]
                # exclude the selected item
                same_orig_indices = same_orig_indices[same_orig_indices != sel_idx]
                similar_df = df.loc[same_orig_indices]
                st.success(f"✅ Found {len(similar_df)} similar items in Cluster {cluster_id}")
                if len(similar_df) == 0:
                    st.info("No other items in this cluster to recommend.")
                else:
                    st.dataframe(similar_df.head(20))
else:
    st.info("👆 Choose an identifier column to enable recommendations.")

st.write("---")
st.caption("Built with ❤️ using Streamlit, KMeans, and PCA")
