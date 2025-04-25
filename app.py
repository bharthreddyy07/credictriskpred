import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("Credit Risk Prediction App")

# Input fields
st.header("Enter Applicant Details")
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.selectbox("Job (Skill Level)", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving Accounts", ["unknown", "little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking Account", ["unknown", "little", "moderate"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)
# Include all Purpose categories from the full dataset
purpose = st.selectbox("Purpose", [
    "radio/TV", "education", "furniture/equipment", "car", "business",
    "domestic appliances", "repairs", "vacation/others"
])

# Preprocess inputs
if st.button("Predict"):
    # Create Age_bin categories
    if age <= 30:
        age_bin = "Young"
    elif age <= 45:
        age_bin = "Middle-aged"
    else:
        age_bin = "Senior"

    # Create input dataframe with all training features
    input_data = pd.DataFrame({
        'Age': [age],
        'Job': [job],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit_per_month': [credit_amount / duration],
        'Sex_male': [1 if sex == "male" else 0],
        'Housing_own': [1 if housing == "own" else 0],
        'Housing_rent': [1 if housing == "rent" else 0],
        'Purpose_car': [1 if purpose == "car" else 0],
        'Purpose_education': [1 if purpose == "education" else 0],
        'Purpose_furniture/equipment': [1 if purpose == "furniture/equipment" else 0],
        'Purpose_radio/TV': [1 if purpose == "radio/TV" else 0],
        'Purpose_domestic appliances': [1 if purpose == "domestic appliances" else 0],
        'Purpose_repairs': [1 if purpose == "repairs" else 0],
        'Purpose_vacation/others': [1 if purpose == "vacation/others" else 0],
        'Age_bin_Middle-aged': [1 if age_bin == "Middle-aged" else 0],
        'Age_bin_Senior': [1 if age_bin == "Senior" else 0]
    })

    # Encode ordinal features
    saving_map = {'unknown': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
    checking_map = {'unknown': 0, 'little': 1, 'moderate': 2}
    input_data['Saving accounts'] = input_data['Saving accounts'].map(saving_map)
    input_data['Checking account'] = input_data['Checking account'].map(checking_map)

    # Scale numerical features
    numerical_cols = ['Age', 'Credit amount', 'Duration', 'Credit_per_month']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Ensure all training features are present
    training_features = model.feature_names_in_  # Get feature names from the trained model
    for feature in training_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Add missing features with 0

    # Reorder columns to match training
    input_data = input_data[training_features]

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader("Prediction")
    st.write("Credit Risk: **Good**" if prediction == 1 else "Credit Risk: **Bad**")
    st.write(f"Probability of Good Credit: {probability:.2%}")