import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("xgb_model.pkl")
imputer = joblib.load("fitted_imputer.joblib")
scaler = joblib.load("fitted_scaler.joblib")

# Sidebar
st.sidebar.title("💧 Water Potability Predictor")
st.sidebar.markdown("""
Built with 💙 by **Rana**  
Helping you make safer water decisions, one prediction at a time.

Upload your water quality data below and get instant insights.
""")

# Main title
st.title("🔬 Is Your Water Safe to Drink?")
st.markdown("""
Welcome to your personal water testing lab.  
Upload a CSV file with water quality parameters, and we’ll predict potability with clarity and care.
""")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your water quality CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    input_df = pd.read_csv(uploaded_file)
    
    # Drop target column if present
    if "Potability" in input_df.columns:
        input_df = input_df.drop(columns=["Potability"])
    
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(input_df)

    try:
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

        # Make predictions readable
        result_df["Potability Prediction"] = result_df["Potability Prediction"].map({
            1: "Potable 💧",
            0: "Not Potable ⚠️"
        })

        st.subheader("🔍 Prediction Results")
        st.markdown("Each row below shows your water sample and its predicted potability. Confidence scores help you interpret the result.")
        st.dataframe(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name="potability_predictions.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("Made with 💙 by Rana. For questions or feedback, feel free to reach out!")

    except Exception as e:
        st.error("Oops! Something went wrong while processing your data. Please check your CSV format and try again.")
        st.exception(e)

else:
    st.info("Please upload a CSV file to begin.")
