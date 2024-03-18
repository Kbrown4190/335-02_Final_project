import streamlit as st
import pandas as pd


# Title of the web app
st.title("Early Detection of Heart Disease Based on Underlying Symptoms")

dataset_path = 'Dataset/Heart.csv'
df = pd.read_csv(dataset_path)
st.write("Here's a preview of the dataset:")
st.dataframe(df.head())

# Read the dataset into DataFrame


