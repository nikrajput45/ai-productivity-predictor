import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="AI Productivity Dashboard", page_icon="📊", layout="wide")

# Load data and model
model = joblib.load("model.pkl")

# Title
st.title("📊 AI Productivity Analytics Dashboard")

# Sidebar inputs
st.sidebar.header("🔮 Predict Productivity")

hours_studied = st.sidebar.slider("📚 Hours Studied", 1, 12, 5)
sleep_hours = st.sidebar.slider("😴 Sleep Hours", 4, 10, 7)
phone_usage = st.sidebar.slider("📱 Phone Usage", 0, 10, 3)

if st.sidebar.button("Predict"):

    input_df = pd.DataFrame(
        [[hours_studied, sleep_hours, phone_usage]],
        columns=["Hours_Studied", "Sleep_Hours", "Phone_Usage"]
    )

    prediction = model.predict(input_df)

    st.sidebar.success(f"🎯 Productivity Score: {prediction[0]:.2f}")

# Layout columns
col1, col2 = st.columns(2)

# Chart 1: Study vs Productivity
with col1:
    st.subheader("📚 Study Hours vs Productivity")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Hours_Studied"], df["Productivity_Score"])
    ax1.set_xlabel("Hours Studied")
    ax1.set_ylabel("Productivity Score")
    st.pyplot(fig1)

# Chart 2: Phone Usage vs Productivity
with col2:
    st.subheader("📱 Phone Usage vs Productivity")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["Phone_Usage"], df["Productivity_Score"])
    ax2.set_xlabel("Phone Usage")
    ax2.set_ylabel("Productivity Score")
    st.pyplot(fig2)

# Correlation Matrix
st.subheader("📈 Correlation Matrix")

fig3, ax3 = plt.subplots()
corr = df.corr()
cax = ax3.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(cax)
st.pyplot(fig3)

# Feature Importance (Model Coefficients)
st.subheader("🧠 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": ["Hours_Studied", "Sleep_Hours", "Phone_Usage"],
    "Importance": model.coef_
})

st.bar_chart(importance_df.set_index("Feature"))

# Data preview
st.subheader("📂 Dataset Preview")

st.dataframe(df.head())
