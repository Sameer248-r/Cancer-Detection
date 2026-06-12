import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Detection Dashboard",
    page_icon="🔬",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #9ea3b8; margin-top: 4px; }
    .winner-badge {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.75rem; color: #a5d6a7; font-weight: 600;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #e0e0e0; margin: 16px 0 8px 0;
        border-left: 4px solid #4fc3f7; padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & preprocess data ────────────────────────────────────────────────────
@st.cache_data
def load_and_train(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # generate synthetic WBC-like data for demo
        np.random.seed(42)
        n = 569
        cols = [
            "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
            "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
            "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
            "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
            "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
            "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
        ]
        X_demo = np.random.randn(n, 30)
        y_demo = (X_demo[:, 0] + X_demo[:, 2] + X_demo[:, 6] + np.random.randn(n)*0.5 > 0.5).astype(int)
        df = pd.DataFrame(X_demo, columns=cols)
        df.insert(0, "diagnosis", ["M" if v else "B" for v in y_demo])
        df.insert(0, "id", range(n))

    # clean
    df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns], inplace=True)
    x = df.loc[:, df.columns[1:]]
    y = df["diagnosis"].map({"M": 1, "B": 0})

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s  = scaler.transform(x_test)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(x_train_s, y_train)
    y_pred_dt = dt.predict(x_test_s)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_s, y_train)
    y_pred_knn = knn.predict(x_test_s)

    metrics = {
        "Decision Tree": {"acc": accuracy_score(y_test, y_pred_dt),
                          "f1":  f1_score(y_test, y_pred_dt),
                          "cm":  confusion_matrix(y_test, y_pred_dt),
                          "y_pred": y_pred_dt,
                          "y_prob": dt.predict_proba(x_test_s)[:, 1]},
        "Logistic Regression": {"acc": accuracy_score(y_test, y_pred_lr),
                                "f1":  f1_score(y_test, y_pred_lr),
                                "cm":  confusion_matrix(y_test, y_pred_lr),
                                "y_pred": y_pred_lr,
                                "y_prob": lr.predict_proba(x_test)[:, 1]},
        "KNN": {"acc": accuracy_score(y_test, y_pred_knn),
                "f1":  f1_score(y_test, y_pred_knn),
                "cm":  confusion_matrix(y_test, y_pred_knn),
                "y_pred": y_pred_knn,
                "y_prob": knn.predict_proba(x_test_s)[:, 1]},
    }

    feat_imp = pd.Series(dt.feature_importances_, index=x.columns).sort_values(ascending=False)

    return df, x, y, x_train, x_test, y_train, y_test, x_train_s, x_test_s, scaler, metrics, feat_imp, dt, lr, knn

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#4fc3f7; margin-bottom:4px;'>🔬 Cancer Detection Dashboard</h1>
<p style='text-align:center; color:#9ea3b8; margin-bottom:24px;'>
    Decision Tree · Logistic Regression · KNN
</p>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown("- 569 patient records\n- 30 features\n- Target: M (Malignant) / B (Benign)")
    st.markdown("---")
    page = st.radio("Navigate", ["📊Overview", "📈 Model Comparison",
                                  "🔍 Confusion Matrix", "📉 ROC Curves",
                                  "🌟 Feature Importance", "🩺 Predict"])

df, x, y, x_train, x_test, y_train, y_test, \
    x_train_s, x_test_s, scaler, metrics, feat_imp, dt, lr, knn = load_and_train(uploaded)

best_model = max(metrics, key=lambda k: metrics[k]["f1"])

# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{x.shape[1]}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with c3:
        mal = (y == 1).sum()
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef5350;">{mal}</div><div class="metric-label">Malignant (M)</div></div>', unsafe_allow_html=True)
    with c4:
        ben = (y == 0).sum()
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#66bb6a;">{ben}</div><div class="metric-label">Benign (B)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        counts = y.value_counts()
        bars = ax.bar(["Benign (B)", "Malignant (M)"], [counts[0], counts[1]],
                      color=["#66bb6a", "#ef5350"], width=0.5, edgecolor="none")
        for bar, val in zip(bars, [counts[0], counts[1]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", color="#9ea3b8")
        ax.tick_params(colors="#9ea3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3f5c")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Feature Correlation Heatmap (Top 10)</div>', unsafe_allow_html=True)
        top_feats = feat_imp.head(10).index.tolist()
        corr = df[top_feats].corr()
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#1e2130")
        ax2.set_facecolor("#1e2130")
        sns.heatmap(corr, ax=ax2, cmap="coolwarm", annot=False,
                    linewidths=0.3, linecolor="#1e2130",
                    cbar_kws={"shrink": 0.7})
        ax2.tick_params(colors="#9ea3b8", labelsize=7)
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="section-header">Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        badge = '<span class="winner-badge">🏆 Best</span>' if name == best_model else ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1rem; color:#ccc; margin-bottom:8px;">{name} {badge}</div>
                <div class="metric-value">{m['acc']*100:.1f}%</div>
                <div class="metric-label">Accuracy</div>
                <hr style="border-color:#3a3f5c; margin:10px 0;">
                <div style="font-size:1.3rem; color:#81d4fa; font-weight:600;">{m['f1']:.4f}</div>
                <div class="metric-label">F1 Score</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1e2130")
    model_names = list(metrics.keys())
    accs  = [metrics[m]["acc"]*100 for m in model_names]
    f1s   = [metrics[m]["f1"] for m in model_names]
    colors = ["#4fc3f7", "#81c784", "#ffb74d"]

    for ax, vals, title, ylab in zip(
        axes,
        [accs, f1s],
        ["Accuracy (%)", "F1 Score"],
        ["Accuracy (%)", "F1 Score"]
    ):
        ax.set_facecolor("#1e2130")
        bars = ax.bar(model_names, vals, color=colors, edgecolor="none", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005 if ylab == "F1 Score" else bar.get_height() + 0.3,
                    f"{v:.3f}" if ylab == "F1 Score" else f"{v:.1f}%",
                    ha="center", color="white", fontsize=10, fontweight="bold")
        ax.set_title(title, color="#e0e0e0", fontsize=12, pad=10)
        ax.set_ylabel(ylab, color="#9ea3b8")
        ax.tick_params(colors="#9ea3b8")
        ax.set_facecolor("#1e2130")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3f5c")
        ymin = min(vals) * 0.95
        ax.set_ylim(ymin, max(vals) * 1.07)

    fig.patch.set_facecolor("#1e2130")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Confusion Matrix":
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#1e2130")
            ax.set_facecolor("#1e2130")
            sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Benign", "Malignant"],
                        yticklabels=["Benign", "Malignant"],
                        ax=ax, linewidths=0.5, linecolor="#1e2130",
                        annot_kws={"size": 14, "weight": "bold"})
            ax.set_xlabel("Predicted", color="#9ea3b8")
            ax.set_ylabel("Actual", color="#9ea3b8")
            ax.tick_params(colors="#9ea3b8")
            fig.patch.set_facecolor("#1e2130")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select Model", list(metrics.keys()))
    report = classification_report(y_test, metrics[sel]["y_pred"],
                                   target_names=["Benign", "Malignant"], output_dict=True)
    rdf = pd.DataFrame(report).transpose()
    st.dataframe(rdf.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 ROC Curves":
    st.markdown('<div class="section-header">ROC Curves — All Models</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    palette = {"Decision Tree": "#4fc3f7", "Logistic Regression": "#81c784", "KNN": "#ffb74d"}
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[name], lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", color="#9ea3b8")
    ax.set_ylabel("True Positive Rate", color="#9ea3b8")
    ax.set_title("ROC Curves Comparison", color="#e0e0e0", fontsize=13)
    ax.legend(facecolor="#252a3d", labelcolor="white", fontsize=10)
    ax.tick_params(colors="#9ea3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f5c")
    fig.patch.set_facecolor("#1e2130")
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">AUC Scores</div>', unsafe_allow_html=True)
    auc_data = {}
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        auc_data[name] = round(auc(fpr, tpr), 4)
    st.dataframe(pd.DataFrame.from_dict(auc_data, orient="index", columns=["AUC Score"]),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌟 Feature Importance":
    st.markdown('<div class="section-header">Decision Tree — Feature Importance</div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N features", 5, 30, 15)
    top = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.4 + 1), facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))[::-1]
    ax.barh(top.index[::-1], top.values[::-1], color=colors_bar, edgecolor="none")
    ax.set_xlabel("Importance", color="#9ea3b8")
    ax.tick_params(colors="#9ea3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f5c")
    ax.set_title("Feature Importance (Decision Tree)", color="#e0e0e0", fontsize=12)
    fig.patch.set_facecolor("#1e2130")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Top Features Table</div>', unsafe_allow_html=True)
    fi_df = feat_imp.head(top_n).reset_index()
    fi_df.columns = ["Feature", "Importance"]
    fi_df["Rank"] = range(1, len(fi_df) + 1)
    st.dataframe(fi_df[["Rank", "Feature", "Importance"]].style.format({"Importance": "{:.4f}"}),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Predict":
    st.markdown('<div class="section-header">Single Patient Prediction</div>', unsafe_allow_html=True)
    st.info("Enter feature values to predict cancer diagnosis.")

    model_choice = st.selectbox("Choose Model", ["Decision Tree", "Logistic Regression", "KNN"])

    feat_cols = x.columns.tolist()
    defaults = x.mean().to_dict()

    st.markdown("**Enter Feature Values (using dataset mean as default):**")
    col1, col2, col3 = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(feat_cols):
        col = [col1, col2, col3][i % 3]
        with col:
            input_vals[feat] = st.number_input(feat, value=float(round(defaults[feat], 4)), format="%.4f")

    if st.button("🔬 Predict Diagnosis", use_container_width=True):
        inp = np.array([[input_vals[f] for f in feat_cols]])

        if model_choice == "Decision Tree":
            inp_scaled = scaler.transform(inp)
            pred = dt.predict(inp_scaled)[0]
            prob = dt.predict_proba(inp_scaled)[0]
        elif model_choice == "Logistic Regression":
            pred = lr.predict(inp)[0]
            prob = lr.predict_proba(inp)[0]
        else:
            inp_scaled = scaler.transform(inp)
            pred = knn.predict(inp_scaled)[0]
            prob = knn.predict_proba(inp_scaled)[0]

        label = "Malignant 🔴" if pred == 1 else "Benign 🟢"
        conf  = prob[pred] * 100

        col_a, col_b = st.columns(2)
        with col_a:
            color = "#ef5350" if pred == 1 else "#66bb6a"
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color};">
                <div class="metric-value" style="color:{color};">{label}</div>
                <div class="metric-label">Predicted Diagnosis</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{conf:.1f}%</div>
                <div class="metric-label">Confidence ({model_choice})</div>
            </div>""", unsafe_allow_html=True)