#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[9]:


import pandas as pd
import numpy as np
import joblib as jb

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("online vs store shopping dataset.csv")

y = df['shopping_preference']

x = df.drop([
    'shopping_preference','impulse_buying_score', 'discount_sensitivity',
    'delivery_fee_sensitivity', 'online_payment_trust_score', 'tech_savvy_score',
    'avg_online_spend', 'environmental_awareness', 'avg_delivery_days',
    'monthly_online_orders', 'time_pressure_level', 'need_touch_feel_score',
    'product_availability_online', 'return_frequency'
], axis=1)

numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = x.select_dtypes(include=['object']).columns.tolist()

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

model = Pipeline([
    ('pre', preprocessor),
    ('reg', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(X_train,y_train)

jb.dump(model,"model.pkl")

print("Model Saved Successfully")
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message":"Shopping Preference Prediction API"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": prediction[0]
    }
import streamlit as st
import requests

st.title("Customer Shopping Preference Prediction")

age = st.number_input('Age')
gender = st.selectbox('Gender', ['Male','Female','Other'])
city_tier = st.selectbox('City Tier', ['Tier 1','Tier 2','Tier 3'])
monthly_income = st.number_input('Monthly Income')
daily_internet_hours = st.number_input('Daily Internet Hours')
smartphone_usage_years = st.number_input('Smartphone Usage Years')
social_media_hours = st.number_input('Social Media Hours')
avg_store_spend = st.number_input('Average Store Spend')
free_return_importance = st.number_input('Free Return Importance')
monthly_store_visits = st.number_input('Monthly Store Visits')
brand_loyalty_score = st.number_input('Brand Loyalty Score')

if st.button("Predict"):

    data = {
        "age":age,
        "gender":gender,
        "city_tier":city_tier,
        "monthly_income":monthly_income,
        "daily_internet_hours":daily_internet_hours,
        "smartphone_usage_years":smartphone_usage_years,
        "social_media_hours":social_media_hours,
        "avg_store_spend":avg_store_spend,
        "free_return_importance":free_return_importance,
        "monthly_store_visits":monthly_store_visits,
        "brand_loyalty_score":brand_loyalty_score
    }

    response = requests.post(
        "https://your-render-url/predict",
        json=data
    )

    st.success(response.json()["prediction"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




