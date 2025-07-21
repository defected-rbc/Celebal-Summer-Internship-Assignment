"""
Streamlit Home Loan Approval Predictor Application

This script creates a web application using Streamlit to interact with
the pre-trained home loan approval prediction model.

To run this application:
1.  Make sure you have 'loan_approval_model.joblib' and 'loan_approval_scaler.joblib'
    files in the same directory as this script.
2.  Install Streamlit: pip install streamlit pandas scikit-learn joblib matplotlib seaborn
3.  Run from your terminal: streamlit run your_app_name.py
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Import Path for robust file handling

# --- 1. Define the Streamlit App Layout (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Home Loan Approval Predictor", layout="centered")

# --- 2. Load the pre-trained model and scaler ---
# Construct paths relative to the current script's directory
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
model_path = current_dir / 'loan_approval_model.joblib'
scaler_path = current_dir / 'loan_approval_scaler.joblib'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found.")
    st.error(f"Attempted to load from: {model_path} and {scaler_path}")
    st.error("Please ensure 'loan_approval_model.joblib' and 'loan_approval_scaler.joblib' are in the same directory as this script.")
    st.stop() # Stop the app if files are not found
except Exception as e:
    st.error(f"An error occurred while loading the model/scaler: {e}")
    st.stop()

st.title("🏡 Home Loan Approval Predictor")
st.markdown("""
    Fill in the details below to predict whether your home loan will be approved or not.
""")

# --- 3. Create Input Fields for User Data ---

# Using columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    st.subheader("Financial & Loan Details")
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount ($k)", min_value=0, value=150)
    loan_amount_term = st.selectbox("Loan Amount Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
    # Changed Credit History input to "Yes" / "No"
    credit_history = st.selectbox("Credit History", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# Define the expected columns (features) in the order the model expects them
# This list must precisely match the columns used during model training after preprocessing.
expected_columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area_Semiurban', 'Property_Area_Urban'
]

# --- 4. Preprocess User Input for Prediction ---
def preprocess_input(gender, married, dependents, education, self_employed,
                     applicant_income, coapplicant_income, loan_amount,
                     loan_amount_term, credit_history, property_area):

    # Create a dictionary to hold the input features
    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history, # Will be mapped below
        'Property_Area_Semiurban': 0, # Initialize for one-hot encoding
        'Property_Area_Urban': 0      # Initialize for one-hot encoding
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Apply Label Encoding (matching the original script's transformation)
    # Gender: Male:1, Female:0
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    # Married: Yes:1, No:0
    input_df['Married'] = input_df['Married'].map({'Yes': 1, 'No': 0})
    # Education: Graduate:0, Not Graduate:1
    input_df['Education'] = input_df['Education'].map({'Graduate': 0, 'Not Graduate': 1})
    # Self_Employed: Yes:1, No:0
    input_df['Self_Employed'] = input_df['Self_Employed'].map({'Yes': 1, 'No': 0})

    # Handle 'Dependents'
    input_df['Dependents'] = input_df['Dependents'].replace('3+', '3').astype(int)

    # Handle Credit_History: Map "Yes" to 1 and "No" to 0
    input_df['Credit_History'] = input_df['Credit_History'].map({'Yes': 1, 'No': 0})


    # Handle One-Hot Encoding for 'Property_Area'
    if property_area == 'Semiurban':
        input_df['Property_Area_Semiurban'] = 1
    elif property_area == 'Urban':
        input_df['Property_Area_Urban'] = 1
    # 'Rural' will have both 'Property_Area_Semiurban' and 'Property_Area_Urban' as 0

    # Select numerical columns for scaling
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    # Apply the loaded StandardScaler
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Reorder columns to match the model's training input
    input_df = input_df[expected_columns]

    return input_df

# --- 5. Make Prediction on Button Click ---
if st.button("Predict Loan Approval"):
    # Preprocess the user's input
    processed_input = preprocess_input(
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    )

    # Make prediction
    prediction_proba = model.predict_proba(processed_input)
    prediction = model.predict(processed_input)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"Loan will likely be **APPROVED** (Probability: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.error(f"Loan will likely be **NOT APPROVED** (Probability: {prediction_proba[0][0]*100:.2f}%)")

    st.markdown("---")

    # --- 6. Visualize Prediction Probabilities ---
    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({
        'Outcome': ['Not Approved', 'Approved'],
        'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
    })

    fig_proba, ax_proba = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Outcome', y='Probability', data=proba_df, palette=['salmon', 'lightgreen'], ax=ax_proba)
    ax_proba.set_ylim(0, 1)
    ax_proba.set_title('Probability of Loan Approval')
    ax_proba.set_ylabel('Probability')
    ax_proba.set_xlabel('')
    for index, row in proba_df.iterrows():
        ax_proba.text(index, row.Probability + 0.02, f'{row.Probability*100:.2f}%', color='black', ha="center")
    st.pyplot(fig_proba)
    plt.close(fig_proba) # Close the figure to prevent display issues

    st.markdown("---")

    # --- 7. Visualize Feature Importance ---
    st.subheader("Feature Importance")
    # Ensure the model has feature importances (e.g., RandomForestClassifier does)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=expected_columns)
        feature_importances = feature_importances.sort_values(ascending=False)

        fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis', ax=ax_feat_imp)
        ax_feat_imp.set_title('Feature Importance for Loan Approval Prediction')
        ax_feat_imp.set_xlabel('Importance Score')
        ax_feat_imp.set_ylabel('Features')
        st.pyplot(fig_feat_imp)
        plt.close(fig_feat_imp) # Close the figure

    else:
        st.info("Feature importance is not available for this model type.")

    st.markdown("---")
    st.write("Input Data Used for Prediction (after preprocessing and scaling):")
    st.dataframe(processed_input)
