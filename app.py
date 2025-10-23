import streamlit as st
import pandas as pd
import joblib

# Load saved model, imputer, and scaler
model = joblib.load("xgb_model.pkl")
imputer = joblib.load("fitted_imputer.joblib")
scaler = joblib.load("fitted_scaler.joblib")

# App title
st.title("💧 Water Potability Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload your water quality CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV
    input_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(input_df)

    # Preprocess: Impute missing values
    imputed_data = imputer.transform(input_df)

    # Scale features
    scaled_data = scaler.transform(imputed_data)

    # Predict
    predictions = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)[:, 1]  # Probability of class 1 (potable)

    # Show results
    result_df = input_df.copy()
    result_df["Potability Prediction"] = predictions
    result_df["Confidence (Potable %)"] = (prediction_proba * 100).round(2)

    st.subheader("🔍 Prediction Results")
    st.write(result_df)

    # Optional: Download results
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results as CSV", data=csv, file_name="potability_predictions.csv", mime="text/csv")