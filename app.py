"""
Customer Churn Prediction Dashboard
Author: Techxaone
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Churn Prediction | Techxaone",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(15,23,42,0.05);
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 999px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    if not os.path.exists('models/churn_model.pkl'):
        return None
    return joblib.load('models/churn_model.pkl')


@st.cache_data
def load_data():
    if not os.path.exists('data/churn_data.csv'):
        return None
    return pd.read_csv('data/churn_data.csv')


# ==================== HEADER ====================
st.markdown('<h1 class="main-header">Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered dashboard to predict customer churn risk in real-time</p>', unsafe_allow_html=True)
st.markdown("---")

# ==================== CHECK FILES ====================
model_data = load_model()
df = load_data()

if model_data is None or df is None:
    st.error("⚠️ Model or data not found. Please run `python train_model.py` first.")
    st.stop()

model = model_data['model']
scaler = model_data['scaler']
encoders = model_data['encoders']
feature_names = model_data['feature_names']
model_name = model_data['model_name']

# ==================== SIDEBAR ====================
st.sidebar.markdown("## 🎯 Customer Input")
st.sidebar.markdown("Enter customer details to predict churn:")

with st.sidebar:
    age = st.slider("Age", 18, 75, 35)
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 70.0)
    total_charges = st.slider("Total Charges ($)", 100.0, 8000.0, 1500.0)
    contract_type = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox("Payment Method", ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No'])
    num_support_calls = st.slider("Support Calls", 0, 10, 2)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True)

# ==================== MAIN LAYOUT ====================
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churn_rate = df['churn'].mean() * 100
avg_tenure = df['tenure_months'].mean()
avg_monthly = df['monthly_charges'].mean()

with col1:
    st.metric("👥 Total Customers", f"{total_customers:,}")
with col2:
    st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
with col3:
    st.metric("⏱️ Avg Tenure", f"{avg_tenure:.0f} mo")
with col4:
    st.metric("💰 Avg Monthly", f"${avg_monthly:.0f}")

st.markdown("---")

# ==================== PREDICTION ====================
if predict_btn:
    input_data = pd.DataFrame({
        'age': [age],
        'tenure_months': [tenure],
        'monthly_charges': [monthly_charges],
        'total_charges': [total_charges],
        'contract_type': [encoders['contract_type'].transform([contract_type])[0]],
        'payment_method': [encoders['payment_method'].transform([payment_method])[0]],
        'internet_service': [encoders['internet_service'].transform([internet_service])[0]],
        'tech_support': [encoders['tech_support'].transform([tech_support])[0]],
        'online_security': [encoders['online_security'].transform([online_security])[0]],
        'num_support_calls': [num_support_calls],
        'satisfaction_score': [satisfaction],
    })

    input_scaled = scaler.transform(input_data[feature_names])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    churn_prob = probability[1] * 100

    st.markdown("### 🎯 Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error(f"### ⚠️ HIGH RISK\n**Churn Probability: {churn_prob:.1f}%**")
        else:
            st.success(f"### ✅ LOW RISK\n**Churn Probability: {churn_prob:.1f}%**")

    with res_col2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk %", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if churn_prob > 50 else "#10b981"},
                'steps': [
                    {'range': [0, 30], 'color': "#d1fae5"},
                    {'range': [30, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#fee2e2"}
                ],
            }
        ))
        gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("---")

# ==================== VISUALIZATIONS ====================
st.markdown("### 📊 Dataset Insights")

tab1, tab2, tab3, tab4 = st.tabs(["Churn Distribution", "Contract Analysis", "Charges vs Churn", "Correlation"])

with tab1:
    fig = px.pie(df, names='churn', title='Churn Distribution',
                 color_discrete_sequence=['#6366f1', '#ef4444'],
                 labels={'churn': 'Churn'})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    contract_churn = df.groupby(['contract_type', 'churn']).size().reset_index(name='count')
    fig = px.bar(contract_churn, x='contract_type', y='count', color='churn',
                 barmode='group', title='Churn by Contract Type',
                 color_discrete_sequence=['#6366f1', '#ef4444'])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.box(df, x='churn', y='monthly_charges', color='churn',
                 title='Monthly Charges vs Churn',
                 color_discrete_sequence=['#6366f1', '#ef4444'])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Feature Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)

# ==================== DATA PREVIEW ====================
st.markdown("---")
st.markdown("### 📋 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8;'>Built with ❤️ by <b>Techxaone</b> — AI & Machine Learning Solutions</p>",
    unsafe_allow_html=True
)