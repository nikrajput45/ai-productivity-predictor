import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Productivity Predictor",
    page_icon="📊",
    layout="centered"
)

# Load trained model
model = joblib.load("model.pkl")

# Title
st.title("📊 AI Productivity Predictor")
st.write("Predict productivity score based on study habits and lifestyle factors.")

st.divider()

# Sidebar Inputs
st.sidebar.header("🔮 Enter Your Daily Data")

hours_studied = st.sidebar.slider("📚 Hours Studied", 1, 12, 5)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 4, 10, 7)
phone_usage = st.sidebar.slider("📱 Phone Usage (hrs)", 0, 10, 3)

# Prediction Button
if st.sidebar.button("Predict Productivity"):

    input_data = pd.DataFrame(
        [[hours_studied, sleep_hours, phone_usage]],
        columns=["Hours_Studied", "Sleep_Hours", "Phone_Usage"]
    )

    prediction = model.predict(input_data)

    st.success(f"🎯 Predicted Productivity Score: {prediction[0]:.2f}")

st.divider()

# Feature Importance Section
st.subheader("🧠 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": ["Hours Studied", "Sleep Hours", "Phone Usage"],
    "Impact on Productivity": model.coef_
})

st.bar_chart(importance_df.set_index("Feature"))

st.caption("Model built using Linear Regression and deployed with Streamlit.")
