"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
chart2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': 'blue'},
            'steps': [
                {'range': [0, 100], 'color': "red"},
                {'range': [100, 200], 'color': "orange"},
                {'range': [200, 300], 'color': "green"}]
            },
        domain={'x': [0, 1], 'y': [0.8, 1]},
        title={'text': "Speed"}
    ))

st.plotly_chart(chart2)+


progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
