import streamlit as st
import pickle
import panda as pd
# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🍽️ AI Restaurant Inventory Forecasting")

day = st.number_input("Enter Day Number", min_value=1)

if st.button("Predict Sales"):
   import pandas as pd

prediction = model.predict(pd.DataFrame([[day]], columns=["Day"]))
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

    tomatoes = prediction[0] * 0.1
    onions = prediction[0] * 0.05
    bread = prediction[0] * 0.2

    st.write(f"🍅 Tomatoes Needed: {tomatoes:.0f} kg")
    st.write(f"🧅 Onions Needed: {onions:.0f} kg")
    st.write(f"🍞 Bread Needed: {bread:.0f} pcs")
