import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "rahulmayank/tourism-wellness-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    repo_type="model",
    filename="tourism_best_model.pkl"
)

model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Predictor", layout="wide")
st.title("Tourism Package Purchase Predictor")
st.write("Predict whether a customer will buy the Wellness Tourism Package.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    numberperson = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferredstar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    numbertrips = st.number_input("Number of Trips", min_value=0, max_value=30, value=3)
    passport = st.selectbox("Passport", [0, 1])
    pitchscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    owncar = st.selectbox("Own Car", [0, 1])
    children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    duration = st.number_input("Duration of Pitch", min_value=1, max_value=200, value=20)
    income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=20000)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": numberperson,
        "NumberOfFollowups": followups,
        "ProductPitched": productpitched,
        "PreferredPropertyStar": preferredstar,
        "MaritalStatus": maritalstatus,
        "NumberOfTrips": numbertrips,
        "Passport": passport,
        "PitchSatisfactionScore": pitchscore,
        "OwnCar": owncar,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "DurationOfPitch": duration,
        "MonthlyIncome": income
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.write("Prediction:", "Will Buy" if pred == 1 else "Will Not Buy")
    st.write(f"Probability of buying: {prob:.2%}")
