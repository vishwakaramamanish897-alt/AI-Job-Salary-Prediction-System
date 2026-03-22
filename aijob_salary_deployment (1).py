import streamlit as st
import pandas as pd
import joblib

# ===== LOAD FILES =====
model = joblib.load("aijob.pkl")
encoder = joblib.load("encoder.pkl")
columns = joblib.load("columns.pkl")

st.title("💼 Salary Prediction System")

# ===== INPUTS (SAFE - using encoder classes) =====
job_title = st.selectbox("Job Title", encoder['job_title'].classes_)
company_size = st.selectbox("Company Size", encoder['company_size'].classes_)
company_industry = st.selectbox("Company Industry", encoder['company_industry'].classes_)
country = st.selectbox("Country", encoder['country'].classes_)

remote_type = st.selectbox("Remote Type", encoder['remote_type'].classes_)
experience_level = st.selectbox("Experience Level", encoder['experience_level'].classes_)
education_level = st.selectbox("Education Level", encoder['education_level'].classes_)

years_experience = st.number_input("Years of Experience", min_value=0)

skills_python = st.selectbox("Python Skill", [0, 1])
skills_sql = st.selectbox("SQL Skill", [0, 1])
skills_ml = st.selectbox("ML Skill", [0, 1])
skills_deep_learning = st.selectbox("Deep Learning", [0, 1])
skills_cloud = st.selectbox("Cloud", [0, 1])

job_posting_month = st.number_input("Month", 1, 12)
job_posting_year = st.number_input("Year", 2000, 2100)

hiring_urgency = st.selectbox("Hiring Urgency", encoder['hiring_urgency'].classes_)
job_openings = st.number_input("Job Openings", min_value=1)

# ===== CREATE DATAFRAME =====
input_df = pd.DataFrame({
    "job_title": [job_title],
    "company_size": [company_size],
    "company_industry": [company_industry],
    "country": [country],
    "remote_type": [remote_type],
    "experience_level": [experience_level],
    "years_experience": [years_experience],
    "education_level": [education_level],
    "skills_python": [skills_python],
    "skills_sql": [skills_sql],
    "skills_ml": [skills_ml],
    "skills_deep_learning": [skills_deep_learning],
    "skills_cloud": [skills_cloud],
    "job_posting_month": [job_posting_month],
    "job_posting_year": [job_posting_year],
    "hiring_urgency": [hiring_urgency],
    "job_openings": [job_openings]
})

# ===== OPTIONAL CLEANING =====
for col in ['job_title', 'company_industry', 'country']:
    input_df[col] = input_df[col].astype(str).str.strip()

# ===== APPLY ENCODING (SAFE) =====
for col in encoder:
    if col in input_df.columns:
        try:
            input_df[col] = encoder[col].transform(input_df[col])
        except ValueError:
            st.error(f"❌ Invalid value in {col}. Please select from dropdown.")
            st.stop()

# ===== MATCH COLUMN ORDER (FIXED) =====
input_df = pd.get_dummies(input_df)

# match with training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# ===== DEBUG (optional) =====
# st.write("Input Data:", input_df)

# ===== PREDICTION =====
if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: {round(prediction, 2)}")
    except Exception as e:
        st.error(f"Error: {e}")
