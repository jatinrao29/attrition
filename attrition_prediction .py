#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("Employee Attrition Prediction App")

# Static file path (remove uploader)
try:
    df = pd.read_csv("Attrition.csv")

    df.fillna(df.drop(columns=["employee_id", "department"]).median(), inplace=True)
    df = pd.get_dummies(df, columns=["department"], drop_first=True)

    if 'attrition' not in df.columns or df['attrition'].nunique() < 2:
        st.error("The data contains only one class for attrition or missing target column.")
    else:
        x = df.drop(columns=["employee_id", "attrition"])
        y = df["attrition"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.07, random_state=42)

        ss = StandardScaler()
        x_train_scaled = ss.fit_transform(x_train)
        x_test_scaled = ss.transform(x_test)

        model = LogisticRegression(penalty='l2', C=1)
        model.fit(x_train_scaled, y_train)
        y_predict = model.predict(x_test_scaled)

        acc = accuracy_score(y_test, y_predict)
        report = classification_report(y_test, y_predict, output_dict=True)
        cm = confusion_matrix(y_test, y_predict)

        st.subheader("Model Evaluation")
        st.write(f"**Accuracy:** {acc:.2f}")

        if '1' in report:
            st.write("**Precision:**", round(report['1']['precision'], 2))
            st.write("**Recall:**", round(report['1']['recall'], 2))
        else:
            st.warning("No positive class (1) found in test predictions. Precision and recall unavailable.")

        st.subheader("Top Predictors of Attrition")
        important_features = [col for col in x.columns if not col.startswith("department_")]
        coef_df = pd.DataFrame({
            'Feature': important_features,
            'Coefficient': model.coef_[0][[x.columns.get_loc(col) for col in important_features]]
        }).sort_values(by='Coefficient', ascending=False)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
        st.pyplot(fig2)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

except FileNotFoundError:
    st.error("`Attrition.csv` file not found in the current directory. Please ensure it exists.")
except Exception as e:
    st.error(f"An error occurred: {e}")

