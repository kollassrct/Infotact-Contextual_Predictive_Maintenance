# ============================================================
# DASHBOARD — Contextual Predictive Maintenance
# Project: Infotact Solutions | Branch: preeti-dev
# Run with: streamlit run dashboard.py
# ============================================================
#
# CHANGELOG (this revision):
#   - Model is now trained ONCE and saved to disk (models/) with joblib.
#     Every subsequent launch loads the saved model + test split instead
#     of retraining from scratch.
#   - LightGBM feature-name fix: feature names are cleaned once, forced
#     onto the training data before fit(), and every downstream frame
#     (test set, noisy test set, SHAP sample) is re-indexed to
#     model.booster_.feature_name() before being passed to the model.
#     This removes the "feature_names mismatch" / silent misalignment
#     risk that noise injection + resampling can otherwise introduce.
#   - Feature importance / SHAP now read feature names directly off the
#     trained booster instead of a separately tracked list, so they can
#     never drift out of sync with the model.
#   - Small UI addition: a "Model Source" badge shows whether the model
#     was loaded from cache or trained fresh (and only trains fresh once).
# ============================================================

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title = "PredictX — Maintenance Intelligence",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0b0f1a; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background-color: #0f1624;
    border-right: 1px solid #1e2d47;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

.header-banner {
    background: linear-gradient(135deg, #0f1f3d 0%, #0a192f 50%, #0d1b2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0070f3, #7c3aed);
}
.header-title {
    font-size: 1.9rem; font-weight: 700;
    color: #f0f6ff; letter-spacing: -0.02em; margin: 0 0 4px 0;
}
.header-sub {
    font-size: 0.82rem; color: #64748b;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
}
.header-badge {
    display: inline-block;
    background: #0d2137; border: 1px solid #1e4976;
    color: #38bdf8; font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px; border-radius: 20px;
    margin-right: 8px; margin-top: 10px;
}
.header-badge-cache {
    display: inline-block;
    background: #0d2b1f; border: 1px solid #1e7645;
    color: #34d399; font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px; border-radius: 20px;
    margin-right: 8px; margin-top: 10px;
}
.header-badge-fresh {
    display: inline-block;
    background: #2b1f0d; border: 1px solid #76551e;
    color: #fbbf24; font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px; border-radius: 20px;
    margin-right: 8px; margin-top: 10px;
}

.kpi-card {
    background: #0f1a2e; border: 1px solid #1e3254;
    border-radius: 10px; padding: 20px 22px;
    text-align: center;
}
.kpi-label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: #475569;
    margin-bottom: 6px; font-weight: 600;
}
.kpi-value {
    font-size: 2rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.kpi-critical { color: #f87171; }
.kpi-warning  { color: #fbbf24; }
.kpi-normal   { color: #34d399; }
.kpi-total    { color: #60a5fa; }
.kpi-avg      { color: #a78bfa; }

.section-header {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #475569;
    font-weight: 700; padding: 0 0 10px 0;
    border-bottom: 1px solid #1e2d47; margin-bottom: 16px;
}

.week-card {
    background: #0f1a2e; border: 1px solid #1e3254;
    border-radius: 10px; padding: 18px 20px; margin-bottom: 10px;
}
.week-title {
    font-size: 0.78rem; font-weight: 700;
    color: #60a5fa; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 8px;
}
.week-item {
    font-size: 0.8rem; color: #94a3b8;
    padding: 3px 0; line-height: 1.6;
}

.noise-card {
    background: #0f1a2e; border: 1px solid #1e3254;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
}

.footer {
    text-align: center; color: #334155;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 24px 0 8px 0; letter-spacing: 0.05em;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Recolor (don't hide) the native header/toolbar so Streamlit's own
   built-in sidebar collapse/expand control keeps working exactly as
   shipped — we only blend its background into the dark theme. */
header[data-testid="stHeader"] { background: #0b0f1a; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────
def clean_col_names(df):
    """Strip characters LightGBM chokes on ([ ] ( ) ,) and normalise
    whitespace to underscores, so feature names are stable, JSON-safe,
    and identical every time this function runs on the same raw columns."""
    df.columns = (
        df.columns
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace(' ', '_', regex=False)
        .str.replace('__', '_', regex=False)
    )
    return df

# ── Saved-model paths ─────────────────────────────────────
MODEL_DIR       = 'models'
MODEL_PATH      = os.path.join(MODEL_DIR, 'lgbm_predictx_model.pkl')
ARTIFACTS_PATH  = os.path.join(MODEL_DIR, 'lgbm_predictx_artifacts.pkl')


# ── Load (or train once + save) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_data():
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data/processed/week2_fused_features.csv')

    drop_cols = ['UDI','Product ID','Type','timestamp','Machine failure',
                 'TWF','HDF','PWF','OSF','RNF']
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols_orig = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols_orig].copy()
    y = df['Machine failure']
    X = clean_col_names(X)
    feature_cols_clean = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    used_cache = os.path.exists(MODEL_PATH) and os.path.exists(ARTIFACTS_PATH)

    if used_cache:
        # ---- Load previously trained model + matching test split ----
        model    = joblib.load(MODEL_PATH)
        artifact = joblib.load(ARTIFACTS_PATH)
        X_test             = artifact['X_test']
        y_test              = artifact['y_test']
        feature_cols_clean  = artifact['feature_cols_clean']
    else:
        # ---- Train fresh (only happens on first-ever launch) ----
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_train, y_train)

        # LightGBM feature-name fix: force the exact same cleaned names
        # onto the resampled training frame and pass them explicitly to
        # fit(), so the names baked into the booster are guaranteed to
        # match feature_cols_clean everywhere downstream.
        X_tr.columns = feature_cols_clean

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=50, scale_pos_weight=(y==0).sum()/(y==1).sum(),
            random_state=42, verbose=-1)
        model.fit(X_tr, y_tr, feature_name=feature_cols_clean)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump({
            'X_test': X_test,
            'y_test': y_test,
            'feature_cols_clean': feature_cols_clean,
        }, ARTIFACTS_PATH)

    # Always re-index to the booster's own feature order/names. This is
    # the safety net that prevents any mismatch (e.g. if a future CSV
    # has columns in a different order) from silently corrupting
    # predictions.
    model_feature_order = model.booster_.feature_name()
    X_test = X_test[model_feature_order]

    return model, df, model_feature_order, feature_cols_orig, X_test, y_test, used_cache


model_source_known_before_load = os.path.exists(MODEL_PATH) and os.path.exists(ARTIFACTS_PATH)
spinner_msg = ("⚡ Loading saved model from cache..." if model_source_known_before_load
               else "⚡ Training model for the first time — this will be cached for future launches...")
with st.spinner(spinner_msg):
    (model, df, feature_cols_clean, feature_cols_orig,
     X_test, y_test, used_cache) = load_model_and_data()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.12em;
color:#334155;font-weight:700;padding-bottom:12px;border-bottom:1px solid #1e2d47;
margin-bottom:16px;">⚙ Control Panel</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATE",
    ["🏠 Overview", "🔍 Fleet Monitor", "📊 Model Performance",
     "📡 Noise Analysis", "🧠 Explainability", "📋 Project Summary"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""<div style="font-size:0.65rem;text-transform:uppercase;
letter-spacing:0.1em;color:#334155;font-weight:700;margin-bottom:10px;">
Prediction Settings</div>""", unsafe_allow_html=True)

alert_threshold = st.sidebar.slider("ALERT THRESHOLD", 0.10, 0.90, 0.50, 0.05)
noise_level     = st.sidebar.selectbox("SENSOR NOISE", [0.00,0.05,0.15,0.30],
    format_func=lambda x:{0.00:"Clean (0%)",0.05:"Low (5%)",
                           0.15:"Medium (15%)",0.30:"High (30%)"}[x])
num_machines    = st.sidebar.slider("MACHINES IN VIEW", 10, 100, 20, 10)

st.sidebar.markdown(f"""
<div style="margin-top:20px;padding-top:16px;border-top:1px solid #1e2d47;">
<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
color:#334155;font-weight:700;margin-bottom:10px;">Dataset Info</div>
<div style="font-size:0.78rem;color:#475569;line-height:2.2;">
📁 <code style="color:#38bdf8;background:#0d2137;padding:1px 6px;
border-radius:4px;">preeti-dev</code><br>
🏢 Infotact Solutions<br>
📊 AI4I 2020 Dataset<br>
🔢 10,000 sensor records<br>
⚠️ Failure rate ~3.4%<br>
🤖 LightGBM + SMOTE<br>
💾 Model: {"loaded from cache" if used_cache else "trained fresh & cached"}
</div></div>""", unsafe_allow_html=True)

if os.path.exists(MODEL_PATH):
    if st.sidebar.button("🔄 Retrain model (clear cache)"):
        for p in (MODEL_PATH, ARTIFACTS_PATH):
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.rerun()

# ── Noise + Predictions ───────────────────────────────────
# feature_cols_clean here IS model.booster_.feature_name(), guaranteed
# in sync with the trained model (see load_model_and_data above).
sensor_clean = [
    c.replace('[','').replace(']','').replace('(','')
     .replace(')','').replace(',','').replace(' ','_').replace('__','_')
    for c in ['Air temperature [K]','Process temperature [K]',
              'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
    if c in feature_cols_orig
]
sensor_clean = [c for c in sensor_clean if c in feature_cols_clean]

def inject_noise(X_data, level):
    np.random.seed(42)
    Xn = X_data.copy()
    for col in sensor_clean:
        if col in Xn.columns:
            Xn[col] = Xn[col] + np.random.normal(0, level*Xn[col].std(), len(Xn))
    return Xn

def align_to_model(X_data):
    """LightGBM feature-name fix: re-index any frame to the exact column
    order/names the booster was trained with before predicting on it."""
    return X_data[model.booster_.feature_name()]

Xd      = inject_noise(X_test, noise_level) if noise_level > 0 else X_test.copy()
Xd      = align_to_model(Xd)
y_proba = model.predict_proba(Xd)[:,1]
y_pred  = (y_proba >= alert_threshold).astype(int)

res = X_test.copy().reset_index(drop=True)
res['prob']      = (y_proba * 100).round(1)
res['predicted'] = y_pred
res['actual']    = y_test.values
res['Machine_ID']= [f"MCH-{i+1:04d}" for i in range(len(res))]
res['status']    = res['prob'].apply(
    lambda p: "CRITICAL" if p >= alert_threshold*100
    else ("WARNING" if p >= alert_threshold*70 else "NORMAL"))

total    = len(res)
critical = (res['status']=='CRITICAL').sum()
warning  = (res['status']=='WARNING').sum()
normal   = (res['status']=='NORMAL').sum()
avg_prob = res['prob'].mean()

# ── Matplotlib theme ──────────────────────────────────────
BG     = '#0f1a2e'
BORDER = '#1e3254'
CR     = '#f87171'   # critical red
CW     = '#fbbf24'   # warning amber
CN     = '#34d399'   # normal green
CA     = '#60a5fa'   # accent blue
CP     = '#a78bfa'   # purple

plt.rcParams.update({
    'figure.facecolor':'#0f1a2e','axes.facecolor':'#0f1a2e',
    'axes.edgecolor':'#1e3254','axes.labelcolor':'#64748b',
    'xtick.color':'#475569','ytick.color':'#475569',
    'text.color':'#94a3b8','grid.color':'#1e2d47',
    'grid.linewidth':0.5,'axes.titlecolor':'#e2e8f0',
    'axes.titleweight':'bold','axes.titlesize':11,'axes.labelsize':9,
})

def status_color(s):
    return CR if s=='CRITICAL' else CW if s=='WARNING' else CN

# ─────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────
if page == "🏠 Overview":
    cache_badge = ('<span class="header-badge-cache">💾 Model: Cached</span>'
                   if used_cache else
                   '<span class="header-badge-fresh">🆕 Model: Freshly Trained</span>')
    st.markdown(f"""
    <div class="header-banner">
        <div class="header-title">⚡ PredictX — Maintenance Intelligence</div>
        <div style="margin-top:6px;">
            <span class="header-badge">preeti-dev</span>
            <span class="header-badge">LightGBM + SMOTE</span>
            <span class="header-badge">IoT Edge AI</span>
            <span class="header-badge">Infotact Solutions</span>
            {cache_badge}
        </div>
        <div class="header-sub" style="margin-top:12px;">
            Contextual Predictive Maintenance · Manufacturing & Automotive · 4-Week Internship
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI Cards
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label, cls in [
        (c1, str(total),         "Total Machines", "kpi-total"),
        (c2, str(critical),      "Critical",       "kpi-critical"),
        (c3, str(warning),       "Warning",        "kpi-warning"),
        (c4, str(normal),        "Normal",         "kpi-normal"),
        (c5, f"{avg_prob:.1f}%", "Avg Risk",       "kpi-avg"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Overview charts row
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown('<div class="section-header">Fleet Health At a Glance</div>', unsafe_allow_html=True)
        sample = res.head(num_machines)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        bar_colors = [status_color(s) for s in sample['status']]
        ax.bar(range(len(sample)), sample['prob'], color=bar_colors, width=0.7, zorder=3)
        ax.axhline(y=alert_threshold*100, color=CR, linestyle='--', linewidth=1.2,
                   alpha=0.8, label=f'Threshold {alert_threshold*100:.0f}%')
        ax.set_xticks(range(len(sample)))
        ax.set_xticklabels(sample['Machine_ID'], rotation=45, ha='right', fontsize=6.5)
        ax.set_ylabel('Failure Probability (%)'); ax.set_ylim(0,115)
        ax.grid(axis='y', zorder=0); ax.set_axisbelow(True)
        ax.legend(handles=[
            mpatches.Patch(color=CR,label='Critical'),
            mpatches.Patch(color=CW,label='Warning'),
            mpatches.Patch(color=CN,label='Normal'),
            plt.Line2D([0],[0],color=CR,linestyle='--',label=f'Threshold ({alert_threshold*100:.0f}%)')
        ], loc='upper right', framealpha=0.15, fontsize=8)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True); plt.close()

    with right:
        st.markdown('<div class="section-header">Status Distribution</div>', unsafe_allow_html=True)
        counts = res['status'].value_counts()
        clrs   = [status_color(s) for s in counts.index]
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        wedges,texts,autotexts = ax2.pie(
            counts.values, labels=counts.index, colors=clrs,
            autopct='%1.1f%%', startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='#0f1a2e', linewidth=2))
        for t in texts: t.set_color('#94a3b8'); t.set_fontsize(9)
        for at in autotexts: at.set_color('#0b0f1a'); at.set_fontsize(8); at.set_fontweight('bold')
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

    # Dataset preview
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Raw Dataset Preview — AI4I 2020</div>', unsafe_allow_html=True)
    preview_cols = [c for c in ['Air temperature [K]','Process temperature [K]',
                       'Rotational speed [rpm]','Torque [Nm]',
                       'Tool wear [min]','Machine failure'] if c in df.columns]
    raw_preview = df[preview_cols].head(8)
    st.dataframe(raw_preview.style.background_gradient(
        subset=['Machine failure'], cmap='Reds') if 'Machine failure' in preview_cols else raw_preview,
        use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────
# PAGE 2 — FLEET MONITOR
# ─────────────────────────────────────────────────────────
elif page == "🔍 Fleet Monitor":
    st.markdown('<div class="section-header">🔍 Fleet Monitor — Machine-Level Risk</div>',
                unsafe_allow_html=True)

    # Risk histogram
    fig3, ax3 = plt.subplots(figsize=(12, 3.2))
    n, bins, patches = ax3.hist(res['prob'], bins=40, edgecolor='#0f1a2e', linewidth=0.4)
    for patch, le in zip(patches, bins[:-1]):
        patch.set_facecolor(CR if le >= alert_threshold*100
                            else CW if le >= alert_threshold*70 else CN)
        patch.set_alpha(0.85)
    ax3.axvline(x=alert_threshold*100, color=CR, linestyle='--',
                linewidth=1.5, label=f'Threshold ({alert_threshold*100:.0f}%)')
    ax3.set_xlabel('Failure Probability (%)'); ax3.set_ylabel('Machine Count')
    ax3.set_title('Risk Score Distribution Across All Machines')
    ax3.legend(fontsize=8, framealpha=0.15); ax3.grid(axis='y')
    plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Critical table
    st.markdown('<div class="section-header">🚨 Critical Machines — Immediate Attention Required</div>',
                unsafe_allow_html=True)

    disp_cols = ['Machine_ID','prob','status','actual'] + \
                [c for c in sensor_clean[:3] if c in res.columns]
    crit_df = (res[res['status']=='CRITICAL']
               .sort_values('prob', ascending=False)[disp_cols]
               .head(30)
               .rename(columns={'prob':'Risk %','status':'Status',
                                'Machine_ID':'Machine ID','actual':'Actual Failure'}))

    if len(crit_df) > 0:
        st.dataframe(
            crit_df.style
                .background_gradient(subset=['Risk %'], cmap='Reds')
                .format({'Risk %':'{:.1f}%'}),
            use_container_width=True, height=400)
    else:
        st.markdown("""<div style="background:#0f2d1f;border:1px solid #064e3b;
        border-radius:8px;padding:16px 20px;color:#34d399;font-size:0.88rem;">
        ✅ No critical machines at this threshold. System healthy.</div>""",
        unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # All machines table
    st.markdown('<div class="section-header">All Machines</div>', unsafe_allow_html=True)
    all_df = res[['Machine_ID','prob','status']].rename(
        columns={'prob':'Risk %','status':'Status','Machine_ID':'Machine ID'})
    st.dataframe(
        all_df.style.background_gradient(subset=['Risk %'], cmap='RdYlGn_r')
              .format({'Risk %':'{:.1f}%'}),
        use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<div class="section-header">📊 Model Performance — LightGBM + SMOTE</div>',
                unsafe_allow_html=True)

    from sklearn.metrics import (confusion_matrix, classification_report,
                                 precision_recall_curve, f1_score,
                                 average_precision_score)

    y_pred_50  = (y_proba >= 0.50).astype(int)
    y_pred_thr = (y_proba >= alert_threshold).astype(int)
    f1_50      = f1_score(y_test, y_pred_50,  average='macro')
    f1_thr     = f1_score(y_test, y_pred_thr, average='macro')
    ap         = average_precision_score(y_test, y_proba)

    # Metric cards
    m1,m2,m3,m4 = st.columns(4)
    for col, val, label, cls in [
        (m1, f"{f1_50:.4f}",  "F1 @ Default (0.50)", "kpi-normal"),
        (m2, f"{f1_thr:.4f}", f"F1 @ Threshold ({alert_threshold:.2f})", "kpi-avg"),
        (m3, f"{ap:.4f}",     "Avg Precision Score",  "kpi-total"),
        (m4, f"{int(y_test.sum())}","True Failures in Test","kpi-critical"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    cl, cr = st.columns(2)

    # Confusion matrix
    with cl:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        import seaborn as sns
        cm = confusion_matrix(y_test, y_pred_thr)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['No Failure','Failure'],
                    yticklabels=['No Failure','Failure'],
                    linewidths=1, linecolor='#0b0f1a',
                    annot_kws={'size':14,'weight':'bold'})
        ax_cm.set_ylabel('Actual'); ax_cm.set_xlabel('Predicted')
        ax_cm.set_title(f'Threshold = {alert_threshold:.2f}')
        plt.tight_layout(); st.pyplot(fig_cm, use_container_width=True); plt.close()

    # PR Curve
    with cr:
        st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
        prec, rec, thr = precision_recall_curve(y_test, y_proba)
        f1_thr_arr = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-9)
        best_idx   = np.argmax(f1_thr_arr)
        fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
        ax_pr.plot(rec, prec, color=CA, linewidth=2, label=f'AP = {ap:.4f}')
        ax_pr.scatter(rec[best_idx], prec[best_idx], color=CR, s=100, zorder=5,
                      label=f'Best threshold = {thr[best_idx]:.2f}')
        ax_pr.fill_between(rec, prec, alpha=0.08, color=CA)
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend(fontsize=8, framealpha=0.15); ax_pr.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_pr, use_container_width=True); plt.close()

    # Classification report
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred_thr,
                                   target_names=['No Failure','Failure'],
                                   output_dict=True)
    rep_df = pd.DataFrame(report).T.round(4)
    st.dataframe(rep_df.style.background_gradient(cmap='Blues', subset=['f1-score']),
                 use_container_width=True)


# ─────────────────────────────────────────────────────────
# PAGE 4 — NOISE ANALYSIS
# ─────────────────────────────────────────────────────────
elif page == "📡 Noise Analysis":
    st.markdown('<div class="section-header">📡 Noise Sensitivity Analysis</div>',
                unsafe_allow_html=True)

    from sklearn.metrics import f1_score as sk_f1

    levels = [0.00,0.05,0.15,0.30]
    labels = ["Clean (0%)","Low (5%)","Medium (15%)","High (30%)"]
    noise_rows = []
    for lvl, lbl in zip(levels, labels):
        Xn   = inject_noise(X_test, lvl) if lvl > 0 else X_test.copy()
        Xn   = align_to_model(Xn)   # LightGBM feature-name fix
        p    = model.predict_proba(Xn)[:,1]
        pred = (p >= alert_threshold).astype(int)
        f1   = sk_f1(y_test, pred, average='macro')
        base = noise_rows[0]['f1'] if noise_rows else f1
        noise_rows.append({'label':lbl,'f1':round(f1,4),
                           'drop':round(base-f1,4),'detected':int(pred.sum())})

    # F1 bar chart
    nl, nr = st.columns([1.5,1])
    with nl:
        fig_n, ax_n = plt.subplots(figsize=(7, 3.8))
        f1s    = [r['f1'] for r in noise_rows]
        bcolors= [CN, CN, CW, CR]
        ax_n.bar(labels, f1s, color=bcolors, width=0.55, zorder=3)
        ax_n.axhline(0.85, color=CR, linestyle='--', linewidth=1.2,
                     alpha=0.8, label='Target F1 = 0.85')
        for i,v in enumerate(f1s):
            ax_n.text(i, v+0.008, f'{v:.4f}', ha='center',
                      fontsize=9, fontweight='600', color='#e2e8f0')
        ax_n.set_ylabel('Macro F1 Score'); ax_n.set_ylim(0,1.05)
        ax_n.set_title('Model F1 Score Under Noise')
        ax_n.grid(axis='y', zorder=0)
        ax_n.legend(fontsize=8, framealpha=0.15)
        plt.tight_layout(); st.pyplot(fig_n, use_container_width=True); plt.close()

    with nr:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        for r in noise_rows:
            color = CR if 'High' in r['label'] else \
                    CW if 'Medium' in r['label'] else CN
            drop_s = f"▼ {r['drop']:.4f}" if r['drop']>0 else "baseline"
            st.markdown(f"""
            <div class="noise-card">
                <div>
                    <div style="font-size:0.72rem;color:#475569;
                    text-transform:uppercase;letter-spacing:0.08em;">
                    {r['label']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                    font-size:1.3rem;font-weight:700;color:{color};">
                    {r['f1']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.75rem;color:#475569;">{drop_s}</div>
                    <div style="font-size:0.78rem;color:#334155;">
                    {r['detected']} flagged</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Signal plots
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Visual Effect of Noise on Air Temperature Sensor</div>',
                unsafe_allow_html=True)

    col_to_plot = sensor_clean[0] if sensor_clean else None
    if col_to_plot:
        fig_sig, axes_sig = plt.subplots(2, 2, figsize=(14, 6))
        axes_sig = axes_sig.flatten()
        x_range  = range(200)
        for i,(lvl,lbl) in enumerate(zip(levels,labels)):
            Xp = inject_noise(X_test, lvl) if lvl > 0 else X_test.copy()
            axes_sig[i].plot(list(x_range), X_test[col_to_plot].values[:200],
                             color=CA, alpha=0.5, linewidth=1, label='Original')
            axes_sig[i].plot(list(x_range), Xp[col_to_plot].values[:200],
                             color=[CN,CN,CW,CR][i], alpha=0.85, linewidth=1,
                             label=f'Noisy ({int(lvl*100)}%)')
            axes_sig[i].set_title(lbl); axes_sig[i].set_xlabel('Reading Index')
            axes_sig[i].set_ylabel(col_to_plot.replace('_',' '))
            axes_sig[i].legend(fontsize=8, framealpha=0.15)
            axes_sig[i].grid(alpha=0.3)
        plt.suptitle('Sensor Signal at Each Noise Level', fontweight='bold', color='#e2e8f0')
        plt.tight_layout(pad=1.5)
        st.pyplot(fig_sig, use_container_width=True); plt.close()


# ─────────────────────────────────────────────────────────
# PAGE 5 — EXPLAINABILITY
# ─────────────────────────────────────────────────────────
elif page == "🧠 Explainability":
    st.markdown('<div class="section-header">🧠 Model Explainability — Feature Intelligence</div>',
                unsafe_allow_html=True)

    ctx_feats = ['ambient_temperature_C','factory_load_density',
                 'temp_gap','load_torque_interaction','heat_stress_index']

    # Feature importance — read names straight off the trained booster
    # (LightGBM feature-name fix) so this can never drift out of sync
    # with what the model actually learned on.
    booster_feature_names = model.booster_.feature_name()
    importances = pd.Series(
        model.feature_importances_, index=booster_feature_names
    ).sort_values(ascending=False).head(15)

    st.markdown('<div class="section-header">Feature Importance — LightGBM</div>',
                unsafe_allow_html=True)
    fig_fi, ax_fi = plt.subplots(figsize=(12, 5))
    colors_fi = [CP if f in ctx_feats else CA for f in importances.index]
    bars_fi   = ax_fi.barh(range(len(importances)), importances.values,
                           color=colors_fi, height=0.65, zorder=3)
    ax_fi.set_yticks(range(len(importances)))
    ax_fi.set_yticklabels(importances.index, fontsize=9)
    ax_fi.invert_yaxis()
    ax_fi.set_xlabel('Importance Score')
    ax_fi.set_title('Top 15 Feature Importances')
    ax_fi.grid(axis='x', zorder=0)
    for bar, val in zip(bars_fi, importances.values):
        ax_fi.text(val + importances.values.max()*0.01,
                   bar.get_y()+bar.get_height()/2,
                   f'{val:.0f}', va='center', fontsize=8, color='#64748b')
    ax_fi.legend(handles=[
        mpatches.Patch(color=CP, label='External Context Feature'),
        mpatches.Patch(color=CA, label='Internal IoT Sensor Feature'),
    ], loc='lower right', framealpha=0.15, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True); plt.close()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # SHAP section
    st.markdown('<div class="section-header">SHAP Analysis</div>', unsafe_allow_html=True)
    try:
        import shap
        with st.spinner("Computing SHAP values..."):
            X_sample    = align_to_model(X_test.sample(n=min(300, len(X_test)), random_state=42))
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

        sl, sr = st.columns(2)
        with sl:
            st.markdown("**SHAP Summary — Beeswarm**", )
            fig_sh1, ax_sh1 = plt.subplots(figsize=(6,5))
            shap.summary_plot(shap_values, X_sample, max_display=12,
                              plot_type='dot', show=False)
            plt.title('SHAP Feature Impact', color='#e2e8f0', fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig_sh1, use_container_width=True); plt.close()

        with sr:
            st.markdown("**SHAP Summary — Bar**")
            fig_sh2, ax_sh2 = plt.subplots(figsize=(6,5))
            shap.summary_plot(shap_values, X_sample, max_display=12,
                              plot_type='bar', show=False)
            plt.title('Mean |SHAP| per Feature', color='#e2e8f0', fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig_sh2, use_container_width=True); plt.close()

        # Waterfall for highest risk machine
        y_proba_sample = model.predict_proba(X_sample)[:,1]
        best_idx = np.argmax(y_proba_sample)
        explanation = shap.Explanation(
            values       = shap_values[best_idx],
            base_values  = explainer.expected_value,
            data         = X_sample.iloc[best_idx].values,
            feature_names= booster_feature_names
        )
        st.markdown(f"**SHAP Waterfall — Highest Risk Machine "
                    f"(Prob: {y_proba_sample[best_idx]*100:.1f}%)**")
        fig_wf, _ = plt.subplots(figsize=(10,5))
        shap.waterfall_plot(explanation, max_display=12, show=False)
        plt.title('Why This Machine is Predicted to Fail',
                  color='#e2e8f0', fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True); plt.close()

    except ImportError:
        st.markdown("""<div style="background:#0d2137;border:1px solid #1e4976;
        border-radius:8px;padding:16px 20px;color:#38bdf8;font-size:0.88rem;">
        ⚠️ SHAP not installed. Run: <code>pip install shap</code>
        then restart the dashboard.</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PAGE 6 — PROJECT SUMMARY
# ─────────────────────────────────────────────────────────
elif page == "📋 Project Summary":
    st.markdown('<div class="section-header">📋 Project Summary — 4-Week Roadmap</div>',
                unsafe_allow_html=True)

    weeks = [
        ("Week 1", "IoT Telemetry Ingestion & Signal Processing", [
            "✅ Loaded AI4I 2020 dataset — 10,000 records, 14 columns",
            "✅ Identified class imbalance — failures = ~3.4% of data",
            "✅ Selected 5 core IoT sensor signals",
            "✅ Computed rolling mean, std, variance (window=10) for all sensors",
            "✅ Generated 15 new rolling feature columns",
            "✅ Produced 5 report charts including correlation heatmap",
        ], CA),
        ("Week 2", "Contextual Data Fusion & Feature Engineering", [
            "✅ Simulated timestamps at 10-minute intervals",
            "✅ Simulated ambient temperature with daily sine-wave cycle",
            "✅ Simulated factory load density with work-hours pattern",
            "✅ Merged internal + external data by timestamp",
            "✅ Created 3 interaction features: temp_gap, load_torque, heat_stress",
            "✅ Ablation study proved external features improve Macro F1",
        ], CP),
        ("Week 3", "Imbalanced Classification & LightGBM Modeling", [
            "✅ Stratified train/test split with SMOTE applied only on the training fold (no leakage)",
            "✅ LightGBM trained with 500 trees, learning_rate=0.05",
            "✅ Model + feature names persisted to disk (models/) — no retraining on every launch",
            "✅ Feature importance + SHAP explainability analysis",
            "✅ Confusion matrix, precision-recall curve generated",
            "✅ Cleaned, LightGBM-safe feature names enforced end-to-end",
        ], CN),
        ("Week 4", "Noise Sensitivity Analysis & Threshold Tuning", [
            "✅ Gaussian noise injected at 4 levels: 0%, 5%, 15%, 30%",
            "✅ Model robustness empirically characterised",
            "✅ Precision-Recall curve plotted",
            "✅ Optimal decision threshold found beyond default 0.50",
            "✅ Side-by-side confusion matrix comparison",
            "✅ Final project summary documented on GitHub",
        ], CW),
    ]

    for wk, title, items, color in weeks:
        st.markdown(f"""
        <div class="week-card" style="border-left: 3px solid {color};">
            <div class="week-title" style="color:{color};">{wk} — {title}</div>
            {''.join(f'<div class="week-item">{i}</div>' for i in items)}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
    t1,t2,t3,t4 = st.columns(4)
    for col, title, items in [
        (t1,"Core ML",["LightGBM","Scikit-learn","imbalanced-learn (SMOTE)","SHAP"]),
        (t2,"Data",["Pandas","NumPy","AI4I 2020 Dataset","Rolling Features"]),
        (t3,"Visualization",["Matplotlib","Seaborn","Streamlit","SHAP Plots"]),
        (t4,"Workflow",["GitHub (preeti-dev)","Jupyter Notebooks","VS Code","Python 3.10+"]),
    ]:
        with col:
            items_html = ''.join(f'<div class="week-item">▸ {i}</div>' for i in items)
            st.markdown(f"""<div class="week-card">
                <div class="week-title">{title}</div>{items_html}
            </div>""", unsafe_allow_html=True)

    # Final metrics row
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Key Results</div>', unsafe_allow_html=True)
    from sklearn.metrics import f1_score as sk_f1
    final_f1 = sk_f1(y_test, (y_proba>=alert_threshold).astype(int), average='macro')
    r1,r2,r3,r4 = st.columns(4)
    for col, val, label, cls in [
        (r1, "10,000",          "Records Processed",   "kpi-total"),
        (r2, "~3.4%",           "Failure Rate",         "kpi-critical"),
        (r3, f"{final_f1:.4f}", "Final Macro F1",       "kpi-normal"),
        (r4, "Cached Model",    "Deployment Mode",      "kpi-avg"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}" style="font-size:1.5rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PREDICTX · CONTEXTUAL PREDICTIVE MAINTENANCE · INFOTACT SOLUTIONS ·
    BUILT BY PREETI · BRANCH preeti-dev
</div>""", unsafe_allow_html=True)