import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
with open('data/xgboost_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load cleaned data
df = pd.read_csv('data/cleaned_churn.csv')

# Load original data for EDA
df_original = pd.read_csv('data/Telco_Customer_Churn_Dataset  (3).csv')

# App title
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
st.markdown("""
    <style>
    ::-webkit-scrollbar {
        width: 2px;
        height: 2px;
    }
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Title color */
    h1 {
        color: #00d4ff;
    }
    
    /* Subheader color */
    h2, h3 {
        color: #a0a0c0;
    }
    
    /* Button */
    div.stButton > button {
        background-color: #00d4ff;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 30px;
        border: none;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background-color: #0099bb;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📁 Dataset Overview",
    "📊 EDA",
    "🧪 Predict Churn"
])

if page == "📁 Dataset Overview":
    st.title("📁 Dataset Overview")

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", df_original.shape[0])
    col2.metric("Total Features", df_original.shape[1])
    col3.metric("Churned Customers", df_original[df_original["Churn"] == "Yes"].shape[0])
    col4.metric("Churn Rate", f"{round(df_original[df_original['Churn'] == 'Yes'].shape[0] / df_original.shape[0] * 100, 2)}%")

    
    # Show dataframe
    st.subheader("Raw Dataset")
    st.dataframe(df_original, use_container_width=True, height=300)

    # Data types
    st.subheader("Column Information")
    st.dataframe(df_original.dtypes.reset_index().rename(columns={0: "dtype", "index": "Column"}), use_container_width=True)

    # Missing values
    st.subheader("Missing Values")
    st.dataframe(df_original.isnull().sum().reset_index().rename(columns={0: "Missing Count", "index": "Column"}), use_container_width=True)    

    # Data types
    st.subheader("Column Information")
    st.dataframe(df_original.dtypes.reset_index().rename(columns={0: "dtype", "index": "Column"}))

    # Missing values
    st.subheader("Missing Values")
    st.dataframe(df_original.isnull().sum().reset_index().rename(columns={0: "Missing Count", "index": "Column"}))

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # Churn Distribution
    st.subheader("Churn Distribution")
    fig = px.pie(df_original, names="Churn", title="Churn vs No Churn",
                 color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig, use_container_width=True)

    # Contract vs Churn
    st.subheader("Contract Type vs Churn")
    fig2 = px.histogram(df_original, x="Contract", color="Churn",
                        barmode="group",
                        color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig2, use_container_width=True)

    # Internet Service vs Churn
    st.subheader("Internet Service vs Churn")
    fig3 = px.histogram(df_original, x="InternetService", color="Churn",
                        barmode="group",
                        color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig3, use_container_width=True)

    # Tenure vs Churn
    st.subheader("Tenure vs Churn")
    fig4 = px.box(df_original, x="Churn", y="tenure",
                  color="Churn",
                  color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig4, use_container_width=True)

    # Monthly Charges vs Churn
    st.subheader("Monthly Charges vs Churn")
    fig5 = px.box(df_original, x="Churn", y="MonthlyCharges",
                  color="Churn",
                  color_discrete_map={"Yes": "red", "No": "green"})
    st.plotly_chart(fig5, use_container_width=True)

    # All Categorical Features vs Churn
    st.subheader("Categorical Features vs Churn")
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    fig6, axes = plt.subplots(4, 4, figsize=(24, 20))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        sns.countplot(
            x=col, hue='Churn', data=df_original,
            palette=['green', 'red'], ax=axes[i]
        )
        axes[i].set_title(f'{col} vs Churn', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=30)
    plt.suptitle('Categorical Features vs Churn', fontsize=20, y=1.02)
    plt.tight_layout()
    st.pyplot(fig6)

elif page == "🧪 Predict Churn":
    st.title("🧪 Predict Customer Churn")
    st.write("Fill in the customer details below to predict if they will churn.")

    col1, col2, col3 = st.columns(3)

    with col1:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges", 18.0, 9000.0, 500.0)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    # Predict button
    if st.button("Predict Churn"):
        input_data = {
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender_Male": 1 if gender == "Male" else 0,
            "Partner_Yes": 1 if partner == "Yes" else 0,
            "Dependents_Yes": 1 if dependents == "Yes" else 0,
            "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
            "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
            "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
            "InternetService_No": 1 if internet_service == "No" else 0,
            "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
            "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
            "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
            "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
            "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
            "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,
            "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ This customer is LIKELY TO CHURN — {round(probability * 100, 2)}% churn risk")
        else:
            st.success(f"✅ This customer is NOT likely to churn — {round(probability * 100, 2)}% churn risk")