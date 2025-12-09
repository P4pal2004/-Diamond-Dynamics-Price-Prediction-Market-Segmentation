# app.py — Fresh Streamlit app for Diamond Dynamics
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# Optional: PyTorch for ANN
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --------- CONFIG ----------
ARTIFACTS_DIR = "artifacts"
st.set_page_config(page_title="Diamond Dynamics", page_icon="💎", layout="wide")
st.title("💎 Diamond Dynamics — Price Prediction & Market Segmentation")

# --------- UTILITIES ----------
def format_price_inr(x):
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return str(x)

def safe_joblib_load(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

# --------- LOAD ARTIFACTS ----------
scaler = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
encoder_dict = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "encoder_dict.pkl"))
model_features = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "model_features.pkl"))
rf_model = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "rf_model.pkl"))
kmeans_model = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl"))
cluster_features = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "kmeans_features.pkl"))
cluster_name_mapping = safe_joblib_load(os.path.join(ARTIFACTS_DIR, "cluster_name_mapping.pkl"))

# Fallbacks
if cluster_name_mapping is None:
    cluster_name_mapping = {0:"Premium Large Diamonds",1:"Mid-Range Diamonds",2:"Budget Small Diamonds",3:"Standard Diamonds"}

sample_csv = os.path.join(ARTIFACTS_DIR, "diamonds_sample.csv")

# --------- ANN Helper ----------
def build_ann(input_dim):
    class SimpleANN(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.layers(x)
    return SimpleANN(input_dim)

ann_path = os.path.join(ARTIFACTS_DIR, "ann_model.pth")
ANN_AVAILABLE = TORCH_AVAILABLE and os.path.exists(ann_path)

# --------- INPUT FORM ----------
st.sidebar.header("Input Diamond Attributes")
carat = st.sidebar.number_input("Carat", 0.01, 10.0, 0.75)
x_dim = st.sidebar.number_input("Length X (mm)", 0.1, 30.0, 5.0)
y_dim = st.sidebar.number_input("Width Y (mm)", 0.1, 30.0, 5.0)
z_dim = st.sidebar.number_input("Depth Z (mm)", 0.1, 30.0, 3.0)
depth = st.sidebar.number_input("Depth (%)", 1.0, 100.0, 61.0)
table = st.sidebar.number_input("Table (%)", 1.0, 100.0, 57.0)

cut_options = ["Ideal","Premium","Very Good","Good","Fair"]
color_options = ["D","E","F","G","H","I","J"]
clarity_options = ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"]

cut = st.sidebar.selectbox("Cut", cut_options)
color = st.sidebar.selectbox("Color", color_options)
clarity = st.sidebar.selectbox("Clarity", clarity_options)

inr_rate = st.sidebar.number_input("Conversion rate USD → INR", 70.0, 100.0, 82.0, 0.01)

uploaded_file = st.sidebar.file_uploader("Batch CSV (optional)", type=["csv"])

# --------- Build input DataFrame ----------
def build_input_df():
    df = pd.DataFrame([{"carat":carat,"x":x_dim,"y":y_dim,"z":z_dim,"depth":depth,"table":table,
                        "cut":cut,"color":color,"clarity":clarity}])
    # log features if in model_features
    if model_features:
        for col in ["carat","depth","table","x","y","z"]:
            target = col+"_log"
            if target in model_features:
                df[target] = np.log1p(df[col])
        # ordinal encode
        for col, opts in zip(["cut","color","clarity"], [cut_options,color_options,clarity_options]):
            target = col+"_ord"
            if target in model_features:
                df[target] = df[col].apply(lambda x: opts.index(x) if x in opts else 0)
        # ensure all features present
        for feat in model_features:
            if feat not in df.columns:
                df[feat]=0
        df = df[model_features].copy()
    return df

input_df = build_input_df()

# --------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Predict","Cluster","EDA","Batch/Export"])

# ---------- PREDICT ----------
with tab1:
    st.header("🎯 Price Prediction Module")
    st.dataframe(input_df.T,width=400)

    col1,col2 = st.columns(2)
    with col1:
        if st.button("🔮 Predict Price (RF)"):
            if rf_model is None or scaler is None:
                st.error("RF model or scaler missing")
            else:
                try:
                    scaled = scaler.transform(input_df)
                    usd_pred = rf_model.predict(scaled)[0]
                    inr_pred = usd_pred*inr_rate
                    st.success(f"💰 RF Predicted Price: {format_price_inr(inr_pred)} (INR) — ${usd_pred:.2f} USD")
                except Exception as e:
                    st.exception(f"RF prediction failed: {e}")

    with col2:
        if st.button("🤖 Predict Price (ANN)"):
            if not ANN_AVAILABLE:
                st.warning("ANN model not available")
            else:
                try:
                    in_dim = len(model_features) if model_features else input_df.shape[1]
                    ann = build_ann(in_dim)
                    ann.load_state_dict(torch.load(ann_path,map_location='cpu'))
                    ann.eval()
                    t = torch.tensor(input_df.values.astype(np.float32))
                    with torch.no_grad():
                        out = ann(t).numpy().reshape(-1)[0]
                    st.success(f"🤖 ANN Predicted Price: {format_price_inr(out*inr_rate)} (INR) — ${out:.2f} USD")
                except Exception as e:
                    st.exception(f"ANN prediction failed: {e}")

# ---------- CLUSTER ----------
with tab2:
    st.header("📊 Market Segment Prediction")
    if st.button("🔍 Predict Cluster"):
        if kmeans_model is None:
            st.error("KMeans model not loaded")
        else:
            try:
                feats = cluster_features if cluster_features else [c for c in model_features if c in input_df.columns]
                feats = [c for c in feats if c in input_df.columns]
                cluster_input = input_df[feats]
                label = int(kmeans_model.predict(cluster_input)[0])
                name = cluster_name_mapping.get(label,f"Cluster {label}")
                st.success(f"Predicted cluster: {label} → {name}")

                # PCA plot with sample
                if os.path.exists(sample_csv):
                    df_sample = pd.read_csv(sample_csv)
                    avail_feats = [c for c in feats if c in df_sample.columns]
                    if len(avail_feats)>=2:
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.decomposition import PCA
                        mat = df_sample[avail_feats].dropna().values
                        pca = PCA(n_components=2)
                        mat_2d = pca.fit_transform(StandardScaler().fit_transform(mat))
                        fig,ax = plt.subplots()
                        ax.scatter(mat_2d[:,0], mat_2d[:,1], alpha=0.3)
                        # input
                        inp = cluster_input[avail_feats].values
                        inp_2d = pca.transform(inp)
                        ax.scatter(inp_2d[:,0], inp_2d[:,1], c='red',marker='X',s=100)
                        ax.set_title("PCA: Input marked RED")
                        st.pyplot(fig)
                    else:
                        st.info("Sample exists but not enough overlapping features for PCA plot.")
            except Exception as e:
                st.exception(f"Cluster prediction failed: {e}")

# ---------- EDA ----------
with tab3:
    st.header("📈 EDA & Sample Preview")
    if os.path.exists(sample_csv):
        df_sample = pd.read_csv(sample_csv)
        st.dataframe(df_sample.head(200))
        if "price" in df_sample.columns:
            fig,ax=plt.subplots()
            ax.hist(df_sample["price"].dropna(),bins=50)
            ax.set_xlabel("Price (USD)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        if {"cut","price"}.issubset(df_sample.columns):
            fig2,ax2=plt.subplots()
            df_sample.boxplot(column="price",by="cut",ax=ax2)
            st.pyplot(fig2)
    else:
        st.info("Place diamonds_sample.csv in artifacts/ for EDA")

# ---------- BATCH ----------
with tab4:
    st.header("📤 Batch Prediction")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df_batch.head())
        try:
            # create model_features if missing
            for col in ["carat","depth","table","x","y","z"]:
                if col+"_log" in model_features and col+"_log" not in df_batch.columns:
                    df_batch[col+"_log"] = np.log1p(df_batch[col])
            for col, opts in zip(["cut","color","clarity"], [cut_options,color_options,clarity_options]):
                if col+"_ord" in model_features and col+"_ord" not in df_batch.columns:
                    df_batch[col+"_ord"] = df_batch[col].apply(lambda x: opts.index(x) if x in opts else 0)
            X_batch = df_batch[model_features]
            X_scaled = scaler.transform(X_batch) if scaler else X_batch.values
            preds_usd = rf_model.predict(X_scaled) if rf_model else np.zeros(len(X_scaled))
            df_batch["pred_price_usd"] = preds_usd
            df_batch["pred_price_inr"] = preds_usd*inr_rate
            st.dataframe(df_batch.head(20))
            # Download
            import io
            buf = io.BytesIO()
            df_batch.to_csv(buf,index=False)
            buf.seek(0)
            st.download_button("⬇ Download CSV",buf,"predictions.csv")
        except Exception as e:
            st.exception(f"Batch prediction failed: {e}")
