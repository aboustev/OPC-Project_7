"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import pandas as pd
import requests
from streamlit_option_menu import option_menu
from Web_func import value_from_colline, create_chart
from page_1 import detailed_page_front

st.set_page_config(
    page_title="Bank loans",
    page_icon="(-.-)",
    layout="wide",
    initial_sidebar_state="expanded"
)

host = 'http://127.0.0.1:5000'
client_id = None

try:
    response = requests.get("{}/load_data".format(host))
except requests.exceptions.ConnectionError:
    host = 'https://share.streamlit.io/aboustev/opc-project_7/main/08_WEBSITE_AND_MODELS/flask_back.py'
    response = requests.get("{}/load_data".format(host))

df_grant_desc, df_all_desc = response.json()
df_grant_desc = pd.read_json(df_grant_desc)
df_all_desc = pd.read_json(df_all_desc)
all_columns = df_all_desc.columns

selected = option_menu(
    None,
    ["Client loan status", "Semi-detailed", "Detailed", 'Custom simulation'],
    icons=['bank2', 'briefcase', "briefcase-fill", 'gear'],
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected == "Detailed":
    client_id = detailed_page_front(host, df_grant_desc, df_all_desc, all_columns, client_id)