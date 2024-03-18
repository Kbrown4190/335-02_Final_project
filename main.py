import streamlit as st
import pandas as pd


# Title of the web app
st.title("Early Detection of Heart Disease Based on Underlying Symptoms")

dataset_path = 'Dataset/Heart.csv'

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Read the dataset into DataFrame
df = pd.read_csv(dataset_path)

