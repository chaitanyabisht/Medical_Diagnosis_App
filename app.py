import streamlit as st
import pandas as pd
import altair as alt

from urllib.error import URLError

import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("lb.pkl", "rb") as f:
    lb = pickle.load(f)

with open("symptoms.pkl", "rb") as f:
    symptoms_df = pickle.load(f)


st.set_page_config(
    page_title="Medical-Diagnosis-App",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("Medical Diagnosis App  :thermometer:")

"""
Are you concerned about your health, we present the one stop easy solution to check for the disease you might be suffering from, at your finger tips. You have to enter the symptoms you are having, based on a decision tree, more symptoms you enter, more is the tree traversed and you get more precise prediction of disease.
So what are you waiting for, try it right now✌🏻

"""

@st.cache
def description(symptom):
    df = pd.read_csv('symptom_Description.csv')
    return df.loc[df['Disease'] == symptom]['Description'].values[0]
def predict(symptoms,lb,model):
    x = lb.transform(symptoms)
    x = sum(x)
    return model.predict([x])[0]



try:
    name = st.text_input('Your Name')
    age = st.text_input('Your Age')
    if (name):
        st.subheader(f"Welcome, {name}")
    
    symptoms = st.multiselect(
        "Choose symptoms", list(symptoms_df['Symptom'])
    )
    """


    """

    if (st.button("Predict")):
        if (symptoms):
            result = predict(symptoms,lb,model)
            st.success(f'Hi {name}, the predicted disease is {result}')
            st.info(description(result))
            col1, col2, col3 = st.columns(3)
            col1.metric("Name", name)
            col2.metric("Age", age)
            col3.metric("Disease", result)
        else:
            st.error("Please select at least one symptom.")
    
    st.subheader("About")
    st.write("This web app is developed by Chaitanya and Ananya for the assignment of DS250 course at IIT Bhilai. The model was generated from the .ipynb notebook.")
except URLError as e:
    st.error(
        """
        **This app requires internet access.**

        Connection error: %s
    """
        % e.reason
    )