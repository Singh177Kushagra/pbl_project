import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

scaler = joblib.load("scaler.pkl")

model_files = {
    "Random Forest": "rf_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "Naive Bayes": "nb_model.pkl"
}

st.title("🔐 IoT Anomaly Detection System (Pretrained Models)")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

selected_model = st.selectbox("Select Model", list(model_files.keys()))

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())


    data.dropna(inplace=True)

    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]

   
    if y_true.dtype == 'object':
        y_true = pd.factorize(y_true)[0]

    
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = pd.factorize(X[col])[0]

    X = X.apply(pd.to_numeric)

    
    X = scaler.transform(X)

    model = joblib.load(model_files[selected_model])

    y_pred = model.predict(X)

    data['Prediction'] = y_pred

    st.subheader("📊 Prediction Results")
    st.write(data.head())

    normal = (y_pred == 0).sum()
    anomaly = (y_pred != 0).sum()

    st.write(f"✅ Normal: {normal}")
    st.write(f"⚠️ Anomalies: {anomaly}")

    misclassified = data[y_pred != y_true]

    st.subheader("❌ Misclassified Records")
    st.write(misclassified.head())

    cm = confusion_matrix(y_true, y_pred)

    st.subheader("📉 Confusion Matrix")
    st.write(cm)

    st.subheader("📈 Prediction Distribution")
    st.bar_chart(pd.Series(y_pred).value_counts())

    st.subheader("📈 Actual Distribution")
    st.bar_chart(pd.Series(y_true).value_counts())

else:
    st.info("Upload a dataset to begin.")
