# 📊 Customer Churn Analysis & Prediction

A machine learning web app built with Streamlit that predicts whether a telecom customer will churn or not.

## 🔗 Live Demo
https://customer-churn-hwk4pihhdwsa7ku3vygphc.streamlit.app/

## 📌 Project Overview
Customer churn is one of the biggest problems in the telecom industry. This project analyzes customer behavior and predicts churn using an XGBoost classification model trained on the IBM Telco Customer Churn dataset.

## 🚀 Features
- **Dataset Overview** — raw data, column info, missing values
- **EDA Dashboard** — interactive charts showing churn patterns
- **Live Churn Predictor** — fill in customer details and get instant churn prediction with probability

## 🛠️ Tech Stack
- Python
- Streamlit
- XGBoost
- Pandas, NumPy
- Plotly
- Scikit-learn

## 📂 Project Structure
customer_churn/
├── app.py
├── data/
│   ├── Telco_Customer_Churn_Dataset  (3).csv   (Raw data)
│   ├── cleaned_churn.csv
│   └── xgboost_churn_model.pkl
└── requirements.txt

## ⚙️ How to Run Locally
git clone https://github.com/GAURAV4478/customer-churn
cd customer_churn
pip install -r requirements.txt
streamlit run app.py

## 📈 Model Performance
- Algorithm: XGBoost Classifier with GridSearchCV tuning
- Dataset: IBM Telco Customer Churn (7,043 customers)
- Key features: Contract type, tenure, monthly charges, internet service

## 🔍 Key Insights
- Month-to-month contract customers churn the most
- Fiber optic internet users have higher churn rate
- New customers (tenure < 9 months) are most at risk
- Electronic check payment users churn more than others

## 👨‍💻 Author
**Gaurav Thakur**
- GitHub: github.com/GAURAV4478
- LinkedIn: linkedin.com/in/gauravthakur7