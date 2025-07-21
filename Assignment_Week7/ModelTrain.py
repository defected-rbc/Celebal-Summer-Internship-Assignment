"""
Home Loan Approval Prediction Model

This script demonstrates how to build a machine learning model to predict
home loan approval based on the provided Kaggle dataset.

Steps included:
1.  Loading the dataset.
2.  Handling missing values.
3.  Encoding categorical features.
4.  Splitting data into training and testing sets.
5.  Training a RandomForestClassifier model.
6.  Evaluating the model's performance.
7.  Providing an example of how to make a prediction.
8.  Saving the trained model to a file.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib # Import joblib for saving/loading models

# --- 1. Load the dataset ---
# Make sure 'loan_sanction_train.csv' is uploaded to your Google Colab environment.
try:
    df = pd.read_csv('loan_sanction_train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'loan_sanction_train.csv' not found.")
    print("Please upload the file to your Google Colab environment (e.g., by dragging it into the file explorer).")
    exit() # Exit if the file is not found

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Info:")
df.info()

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# --- 2. Data Preprocessing ---

# Drop 'Loan_ID' as it's just an identifier and not useful for prediction
df = df.drop('Loan_ID', axis=1)

# Handle missing values

# Check for missing values
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Impute missing values for numerical columns with the median
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Impute missing values for categorical columns with the mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Encode categorical features

# 'Dependents' column has '3+' which needs to be handled
# Replace '3+' with '3' and convert to numeric
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)

# Apply Label Encoding for binary categorical features and the target variable
le = LabelEncoder()

# Gender, Married, Education, Self_Employed, Loan_Status
df['Gender'] = le.fit_transform(df['Gender']) # Male:1, Female:0
df['Married'] = le.fit_transform(df['Married']) # Yes:1, No:0
df['Education'] = le.fit_transform(df['Education']) # Graduate:0, Not Graduate:1
df['Self_Employed'] = le.fit_transform(df['Self_Employed']) # Yes:1, No:0
df['Loan_Status'] = le.fit_transform(df['Loan_Status']) # Y:1, N:0

# One-hot encode 'Property_Area' as it has more than two categories and no inherent order
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True) # drop_first avoids multicollinearity

# Display the processed data info
print("\nDataset Info after preprocessing:")
df.info()
print("\nFirst 5 rows of the preprocessed dataset:")
print(df.head())

# --- 3. Feature Scaling (for numerical features) ---
# It's good practice to scale numerical features, especially for models sensitive to feature scales.
# RandomForest is less sensitive, but it's a good general step.

numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nFirst 5 rows after feature scaling:")
print(df.head())

# --- 4. Prepare data for modeling ---
X = df.drop('Loan_Status', axis=1) # Features
y = df['Loan_Status'] # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 5. Model Selection and Training ---
# Using RandomForestClassifier, a robust and widely used ensemble model.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("\nTraining the RandomForestClassifier model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Model Evaluation ---
print("\nEvaluating the model...")
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Approved (N)', 'Approved (Y)']))

# Display Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Approved', 'Predicted Approved'],
            yticklabels=['Actual Not Approved', 'Actual Approved'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- 7. Make a Prediction on New Data (Example) ---
print("\n--- Example Prediction ---")

sample_data = {
    'Gender': 1,  # Male (encoded from 'Male')
    'Married': 1, # Yes (encoded from 'Yes')
    'Dependents': 1, # 1 dependent (encoded from '1')
    'Education': 0, # Graduate (encoded from 'Graduate')
    'Self_Employed': 1, # Yes (encoded from 'Yes')
    'ApplicantIncome': 6000,
    'CoapplicantIncome': 1000,
    'LoanAmount': 140,
    'Loan_Amount_Term': 360,
    'Credit_History': 1, # Good credit history
    'Property_Area_Semiurban': 1, # Semiurban (encoded from 'Semiurban')
    'Property_Area_Urban': 0 # Not Urban
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_data])

# Apply the same scaling to numerical features as done on the training data
sample_df[numerical_cols] = scaler.transform(sample_df[numerical_cols])

print("\nSample data for prediction (after scaling):")
print(sample_df)

# Make prediction
prediction_proba = model.predict_proba(sample_df)
prediction = model.predict(sample_df)

print(f"\nPrediction Probability (Not Approved, Approved): {prediction_proba[0]}")

if prediction[0] == 1:
    print("Prediction: Loan will likely be APPROVED (Y)")
else:
    print("Prediction: Loan will likely be NOT APPROVED (N)")

print("\n--- End of Prediction Example ---")

# --- 8. Save the trained model ---
# It's crucial to save the trained model along with the scaler,
# as the scaler is needed to preprocess new data before prediction.
model_filename = 'loan_approval_model.joblib'
scaler_filename = 'loan_approval_scaler.joblib'

try:
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"\nModel saved successfully as '{model_filename}'")
    print(f"Scaler saved successfully as '{scaler_filename}'")
except Exception as e:
    print(f"\nError saving model or scaler: {e}")

