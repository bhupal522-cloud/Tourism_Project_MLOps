import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="bhupal522/Tourism-Customer-Prediction", filename="best_tourism_customer_prediction_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Customer Prediction
st.title("Toursim Customer Prediction App")
st.write("""
This application predicts the likelihood of a customer buying a tourism package based on its operational parameters.
Please enter customer info and interaction data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=75, value=30)
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=12, value=6)
Occupation = st.selectbox("Occupation", ["Self_Employed", "Salaried", "Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=4)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=10, value=2)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=4)
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=4, value=2)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
TypeofContact = st.selectbox("Type of Contact", ["Self_Employed", "Family", "Friends"])
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional"])
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar':OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Designation': Designation,
    'ProductPitched': ProductPitched

    }])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Customer Purchage a Package" if prediction == 1 else "Customer not purchase a Package "
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
