import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
if "history" not in st.session_state:
    st.session_state.history = []
st.title("🍽 AI Restaurant Inventory Forecasting")

day = st.number_input("Enter Day Number", min_value=1)
temperature = st.number_input("Enter Temperature", value=30)
weekend = st.selectbox("Weekend", [0, 1])

if st.button("Predict Sales"):

    input_data = pd.DataFrame(
        [[day, temperature, weekend]],
        columns=["Day", "Temperature", "Weekend"]
    )

    prediction = model.predict(input_data)
    st.session_state.history.append({
    "Day": day,
    "Temperature": temperature,
    "Weekend": weekend,
    "Predicted Sales": round(prediction[0], 2)
})
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

    tomatoes = prediction[0] * 0.1
    onions = prediction[0] * 0.05
    bread = prediction[0] * 0.2

    st.write(f"🍅 Tomatoes Needed: {tomatoes:.0f} kg")
    st.write(f"🧅 Onions Needed: {onions:.0f} kg")
    st.write(f"🍞 Bread Needed: {bread:.0f} pcs")
    st.subheader("Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
# Sales Trend Graph
st.subheader("Sales Trend")

st.line_chart(
    history_df.set_index("Day")["Predicted Sales"]
)
