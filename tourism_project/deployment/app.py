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

# Define mappings for categorical features
occupation_mapping = {"Business": 0, "Salaried": 1, "Self_Employed": 2}
gender_mapping = {"Female": 0, "Male": 1}
marital_status_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
passport_mapping = {"No": 0, "Yes": 1}
own_car_mapping = {"No": 0, "Yes": 1}
type_of_contact_mapping = {"Family": 0, "Friends": 1, "Self_Employed": 2}
designation_mapping = {"Executive": 0, "Managerial": 1, "Professional": 2}
product_pitched_mapping = {"Basic": 0, "Deluxe": 1, "Standard": 2}

# Assemble input into DataFrame with numerical categorical values
input_data = pd.DataFrame([
    {
        'Age': Age,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': occupation_mapping[Occupation],
        'Gender': gender_mapping[Gender],
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': marital_status_mapping[MaritalStatus],
        'NumberOfTrips': NumberOfTrips,
        'Passport': passport_mapping[Passport],
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': own_car_mapping[OwnCar],
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'MonthlyIncome': MonthlyIncome,
        'TypeofContact': type_of_contact_mapping[TypeofContact],
        'Designation': designation_mapping[Designation],
        'ProductPitched': product_pitched_mapping[ProductPitched]
    }
])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Customer Purchage a Package" if prediction == 1 else "Customer not purchase a Package "
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
