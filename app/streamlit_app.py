"""Small web interface to test the model.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import predict_risk  # noqa: E402

st.set_page_config(page_title="Machine failure risk", page_icon="⚙️")
st.title("Machine failure risk")
st.caption("Enter the sensor readings of a production cycle to estimate the risk of failure.")

col1, col2 = st.columns(2)
with col1:
    product_type = st.selectbox("Product quality type", ["L", "M", "H"])
    air = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    process = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
with col2:
    rpm = st.slider("Rotational speed (rpm)", 1150, 2900, 1500, 10)
    torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    wear = st.slider("Tool wear (min)", 0, 255, 100, 1)

proba, alert = predict_risk(product_type, air, process, rpm, torque, wear)

st.metric("Failure probability", f"{proba:.1%}")
if alert:
    st.error("Alert: schedule a preventive inspection before the next cycle.")
else:
    st.success("OK: no action needed.")
