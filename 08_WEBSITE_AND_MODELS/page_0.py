"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
from PIL import Image
import plotly.graph_objects as go


def loan_status_page_front(proba, client_info, client_id=None):
    proba *= 100
    st.write("This page gives an overview of client {}'s loan status".format(client_id))
    if client_info.TARGET_PROBA[0]:
        st.write("Predicted status = {}".format("Granted" if client_info.TARGET[0] == 1 else "Not Granted"))
        chart = go.Figure(go.Indicator(
            mode="gauge+number",
            value=client_info.TARGET_PROBA[0]*100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'blue'},
                'steps': [
                    {'range': [0, proba], 'color': 'green'},
                    {'range': [proba, 100], 'color': 'red'}]
            },
            domain={'x': [0, 1], 'y': [0.4, 1]},
            title={'text': 'Probability of not repaying the loan'}
        ))
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.write("Status = {}".format("Granted" if client_info.TARGET[0] == 1 else "Not Granted"))
    img = Image.open('featureimportance.png')
    col1, col2, col3 = st.columns([20, 60, 20])
    with col2:
        st.image(img, caption='Feature importance')
