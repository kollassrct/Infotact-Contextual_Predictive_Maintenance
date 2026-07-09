# ============================================================
# DASHBOARD — Contextual Predictive Maintenance
# Project: Infotact Solutions | Branch: preeti-dev
# Run with: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title = "Predictive Maintenance Dashboard",
    page_icon  = "🔧",
    layout     = "wide"
)

# ── Title ─────────────────────────────────────────────────
st.title("🔧 Contextual Predictive Maintenance Dashboard")
st.markdown("**Infotact Solutions** | IoT Edge AI | Branch: `preeti-dev`")
st.markdown("---")

# ── Load & Train Model (cached so it only runs once) ──────
@st.cache_resource
def load_model_and_data():
    import lightgbm as lgb
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('data/processed/week2_fused_features.csv')

    drop_cols = ['UDI', 'Product ID', 'Type', 'timestamp',
                 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=50, scale_pos_weight=(y==0).sum()/(y==1).sum(),
        random_state=42, verbose=-1
    )
    model.fit(X_train_sm, y_train_sm)

    return model, df, feature_cols, X_test, y_test

with st.spinner("🔄 Loading model and data... please wait"):
    model, df, feature_cols, X_test, y_test = load_model_and_data()

st.success("✅ Model loaded and ready!")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.markdown("Use the controls below to explore predictions.")

alert_threshold = st.sidebar.slider(
    "🎚️ Alert Threshold",
    min_value = 0.10,
    max_value = 0.90,
    value     = 0.50,
    step      = 0.05,
    help      = "Machines with failure probability above this value will be flagged."
)

noise_level = st.sidebar.selectbox(
    "📡 Sensor Noise Level",
    options = [0.00, 0.05, 0.15, 0.30],
    format_func = lambda x: {
        0.00: "Clean (0%)",
        0.05: "Low (5%)",
        0.15: "Medium (15%)",
        0.30: "High (30%)"
    }[x]
)

num_machines = st.sidebar.slider(
    "🏭 Number of Machines to Show",
    min_value = 10,
    max_value = 100,
    value     = 20,
    step      = 10
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("📁 Branch: `preeti-dev`")
st.sidebar.markdown("🏢 Infotact Solutions")
st.sidebar.markdown("📅 4-Week IoT Edge AI Project")

# ── Apply Noise ───────────────────────────────────────────
sensor_cols_present = [c for c in [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
] if c in X_test.columns]

def inject_noise(X_data, level):
    np.random.seed(42)
    X_noisy = X_data.copy()
    for col in sensor_cols_present:
        noise = np.random.normal(0, level * X_data[col].std(), size=len(X_data))
        X_noisy[col] = X_data[col] + noise
    return X_noisy

X_display = inject_noise(X_test, noise_level) if noise_level > 0 else X_test.copy()

# ── Get Predictions ───────────────────────────────────────
y_proba  = model.predict_proba(X_display)[:, 1]
y_pred   = (y_proba >= alert_threshold).astype(int)

results  = X_display.copy().reset_index(drop=True)
results['Failure_Probability'] = (y_proba * 100).round(1)
results['Predicted']           = y_pred
results['Actual']              = y_test.values
results['Status'] = results['Failure_Probability'].apply(
    lambda p: "🔴 CRITICAL" if p >= alert_threshold * 100
    else ("🟡 WARNING" if p >= (alert_threshold * 100 * 0.7)
    else "🟢 NORMAL")
)
results['Machine_ID'] = [f"MCH-{i+1:04d}" for i in range(len(results))]

# ── KPI Cards Row ─────────────────────────────────────────
total     = len(results)
critical  = (results['Status'] == '🔴 CRITICAL').sum()
warning   = (results['Status'] == '🟡 WARNING').sum()
normal    = (results['Status'] == '🟢 NORMAL').sum()
avg_prob  = results['Failure_Probability'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🏭 Total Machines",  total)
col2.metric("🔴 Critical",        critical,  delta=f"{critical/total*100:.1f}%", delta_color="inverse")
col3.metric("🟡 Warning",         warning)
col4.metric("🟢 Normal",          normal)
col5.metric("📊 Avg Risk Score",  f"{avg_prob:.1f}%")

st.markdown("---")

# ── Fleet Risk Overview Chart ─────────────────────────────
st.subheader("📊 Fleet Risk Overview")

sample = results.head(num_machines).copy()
colors_map = {
    '🔴 CRITICAL': '#e74c3c',
    '🟡 WARNING' : '#f39c12',
    '🟢 NORMAL'  : '#2ecc71'
}
bar_colors = [colors_map[s] for s in sample['Status']]

fig, ax = plt.subplots(figsize=(14, 4))
bars = ax.bar(sample['Machine_ID'], sample['Failure_Probability'],
              color=bar_colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=alert_threshold * 100, color='red', linestyle='--',
           linewidth=1.5, label=f'Alert Threshold ({alert_threshold*100:.0f}%)')
ax.set_xlabel('Machine ID')
ax.set_ylabel('Failure Probability (%)')
ax.set_title(f'Failure Probability — First {num_machines} Machines', fontweight='bold')
ax.set_ylim(0, 110)
plt.xticks(rotation=45, ha='right', fontsize=7)
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Two Column Layout ─────────────────────────────────────
left, right = st.columns(2)

# ── Status Distribution Pie ───────────────────────────────
with left:
    st.subheader("🥧 Fleet Status Distribution")
    status_counts = results['Status'].value_counts()
    pie_colors = [colors_map.get(s, '#95a5a6') for s in status_counts.index]

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.pie(status_counts.values, labels=status_counts.index,
            colors=pie_colors, autopct='%1.1f%%',
            startangle=90, explode=[0.05]*len(status_counts))
    ax2.set_title('Machine Status Breakdown', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Probability Distribution ──────────────────────────────
with right:
    st.subheader("📈 Risk Score Distribution")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.hist(results['Failure_Probability'], bins=40,
             color='steelblue', edgecolor='white', alpha=0.85)
    ax3.axvline(x=alert_threshold * 100, color='red', linestyle='--',
                linewidth=2, label=f'Threshold ({alert_threshold*100:.0f}%)')
    ax3.set_xlabel('Failure Probability (%)')
    ax3.set_ylabel('Number of Machines')
    ax3.set_title('Distribution of Risk Scores', fontweight='bold')
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

st.markdown("---")

# ── Critical Machines Table ───────────────────────────────
st.subheader("🚨 Machines Requiring Immediate Attention")

critical_machines = results[results['Status'] == '🔴 CRITICAL'].sort_values(
    'Failure_Probability', ascending=False
)[['Machine_ID', 'Failure_Probability', 'Status',
   'Tool wear [min]', 'Torque [Nm]', 'Rotational speed [rpm]']].head(20)

if len(critical_machines) > 0:
    st.dataframe(
        critical_machines.style
            .background_gradient(subset=['Failure_Probability'], cmap='Reds')
            .format({'Failure_Probability': '{:.1f}%',
                     'Tool wear [min]'    : '{:.0f}',
                     'Torque [Nm]'        : '{:.1f}',
                     'Rotational speed [rpm]': '{:.0f}'}),
        use_container_width=True,
        height=400
    )
else:
    st.success("✅ No critical machines at this threshold level.")

st.markdown("---")

# ── Feature Importance ────────────────────────────────────
st.subheader("🔍 What's Driving the Predictions?")

importances = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False).head(12)

context_feats = ['ambient_temperature_C', 'factory_load_density',
                 'temp_gap', 'load_torque_interaction', 'heat_stress_index']

fig4, ax4 = plt.subplots(figsize=(10, 5))
colors_fi = ['#e74c3c' if f in context_feats else '#3498db'
             for f in importances.index]
importances.plot(kind='barh', ax=ax4, color=colors_fi)
ax4.invert_yaxis()
ax4.set_title('Top 12 Feature Importances\n(🔴 Red = External Context Feature | 🔵 Blue = Internal Sensor)',
              fontweight='bold')
ax4.set_xlabel('Importance Score')
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.markdown("---")

# ── Noise Impact Summary ──────────────────────────────────
st.subheader("📡 Noise Impact on Model Performance")

from sklearn.metrics import f1_score as sk_f1

noise_data = []
for lvl, lbl in [(0.00,"Clean (0%)"),(0.05,"Low (5%)"),(0.15,"Medium (15%)"),(0.30,"High (30%)")]:
    X_n = inject_noise(X_test, lvl) if lvl > 0 else X_test.copy()
    p   = model.predict_proba(X_n)[:,1]
    pr  = (p >= alert_threshold).astype(int)
    f1  = sk_f1(y_test, pr, average='macro')
    noise_data.append({'Noise Level': lbl, 'Macro F1 Score': round(f1, 4),
                       'Failures Detected': int(pr.sum())})

noise_df = pd.DataFrame(noise_data)
st.dataframe(noise_df, use_container_width=True)

st.markdown("---")
st.markdown(
    "**Built by Preeti** | Infotact Solutions Internship | "
    "Contextual Predictive Maintenance using IoT Edge AI | `preeti-dev`"
)