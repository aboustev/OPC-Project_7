"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import pandas as pd
import requests
from Web_func import value_from_colline, create_chart

st.set_page_config(
    page_title="Bank loans",
    page_icon="(-.-)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

data = pd.read_csv(r'06_MODEL\all_data.csv')
data_granted_loan = data[data.TARGET.values == 1]
columns = data.columns[2:]
granted_described = data_granted_loan[columns].describe()
all_described = data[columns].describe()

st.header('Loan visualization')

with st.sidebar:
    number = st.number_input('Client ID',
                             min_value=data.SK_ID_CURR.min(),
                             max_value=data.SK_ID_CURR.max(),
                             step=1)
    response = requests.get("http://127.0.0.1:5000/api/getdecision/?id={}".format(int(number)))
    st.write(response.json())

col1, col2 = st.columns([20, 20])
columns = [col for col in columns if col != 'TARGET']
num = 0
for col in columns:
    if col == 'AMT_CREDIT':
        color_list = ['red', 'orange', '#4ee44e', 'yellow', '#FFCC00', 'red']
    else:
        color_list = ['red', 'orange', '#FFCC00', 'yellow', '#4ee44e', 'yellow']
    chart = create_chart(all_described,
                         granted_described,
                         color_list,
                         col
                         )
    chart.update_traces(value=value_from_colline(number, col, data))
    if num % 2 == 0:
        with col1:
            st.plotly_chart(chart, use_container_width=True)
    else:
        with col2:
            st.plotly_chart(chart, use_container_width=True)
    num += 1
