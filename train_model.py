"""
Customer Churn Prediction - Model Training Script
Author: Techxaone
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== GENERATE SYNTHETIC DATA ====================
def generate_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 75, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.round(np.random.uniform(20, 120, n_samples), 2),
        'total_charges': np.round(np.random.uniform(100, 8000, n_samples), 2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20]),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'num_support_calls': np.random.randint(0, 10, n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples),
    }

    df = pd.DataFrame(data)

    # 🔥 Stronger, more deterministic churn logic
    score = np.zeros(n_samples)

    # Contract type (biggest factor)
    score += np.where(df['contract_type'] == 'Month-to-month', 2.5, 0)
    score += np.where(df['contract_type'] == 'One year', 0.5, 0)

    # Tenure (shorter = more likely to churn)
    score += np.where(df['tenure_months'] < 6, 2.0, 0)
    score += np.where((df['tenure_months'] >= 6) & (df['tenure_months'] < 12), 1.2, 0)

    # Satisfaction (lowest = churn)
    score += np.where(df['satisfaction_score'] == 1, 2.5, 0)
    score += np.where(df['satisfaction_score'] == 2, 1.5, 0)
    score += np.where(df['satisfaction_score'] >= 4, -1.5, 0)

    # Support calls (many = frustrated)
    score += np.where(df['num_support_calls'] >= 5, 1.8, 0)
    score += np.where(df['num_support_calls'] >= 7, 1.0, 0)

    # High monthly charges
    score += np.where(df['monthly_charges'] > 90, 1.0, 0)

    # No tech support / security
    score += np.where(df['tech_support'] == 'No', 0.8, 0)
    score += np.where(df['online_security'] == 'No', 0.5, 0)

    # Fiber optic users churn more (common in real data)
    score += np.where(df['internet_service'] == 'Fiber optic', 0.7, 0)

    # Electronic check users churn more
    score += np.where(df['payment_method'] == 'Electronic check', 0.5, 0)

    # Convert score to probability using sigmoid
    churn_prob = 1 / (1 + np.exp(-(score - 3.5)))
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)

    return df

# ==================== PREPROCESSING ====================
def preprocess(df):
    df = df.copy()
    df.drop('customer_id', axis=1, inplace=True)

    categorical_cols = ['contract_type', 'payment_method', 'internet_service', 'tech_support', 'online_security']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# ==================== TRAINING ====================
def train():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("📊 Generating dataset...")
    df = generate_dataset()
    df.to_csv('data/churn_data.csv', index=False)
    print(f"✅ Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n🔧 Preprocessing...")
    df_processed, encoders = preprocess(df)

    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🤖 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    # Compare
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)

    for name, pred in [('Random Forest', rf_pred), ('XGBoost', xgb_pred)]:
        print(f"\n{name}:")
        print(f"  Accuracy : {accuracy_score(y_test, pred):.4f}")
        print(f"  Precision: {precision_score(y_test, pred):.4f}")
        print(f"  Recall   : {recall_score(y_test, pred):.4f}")
        print(f"  F1-Score : {f1_score(y_test, pred):.4f}")

    # Save best model
    best_model = xgb_model if f1_score(y_test, xgb_pred) > f1_score(y_test, rf_pred) else rf
    best_name = 'XGBoost' if best_model == xgb_model else 'Random Forest'

    print(f"\n✅ Best model: {best_name}")

    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': list(X.columns),
        'model_name': best_name
    }, 'models/churn_model.pkl')

    print("💾 Model saved: models/churn_model.pkl")
    print("\n🎉 Training complete! Now run: streamlit run app.py")


if __name__ == '__main__':
    train()