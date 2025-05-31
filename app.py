import pandas as pd
import streamlit as st

st.title("Would you have survived the Titanic disaster ?")
st.subheader("This model predicts if he/she survives the titanic disaster")

train_df = pd.read_csv('/Users/vbajoria/desktop/titanic-train.csv')
st.dataframe(train_df.head())





