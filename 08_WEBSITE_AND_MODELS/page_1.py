"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
from Web_func import create_chart


def detailed_page_front(client_info, granted_described, all_described, columns, client_id=None):

    col1, col2, col3 = st.columns([20, 20, 20])
    columns = [col for col in columns if col not in ['SK_ID_CURR', 'TARGET', 'TARGET_PROBA']]
    num = 0
    for col in columns:
        if col == 'AMT_CREDIT':
            color_list = ['red', 'orange', '#4ee44e', 'yellow', '#FFCC00', 'red']
        else:
            color_list = ['red', 'orange', '#FFCC00', 'yellow', '#4ee44e', 'yellow']
        chart = create_chart(all_described,
                             granted_described,
                             color_list,
                             col)
        chart.update_traces(value=client_info.loc['0', col])
        if num % 3 == 0:
            with col1:
                st.plotly_chart(chart, use_container_width=True)
        elif num % 3 == 1:
            with col2:
                st.plotly_chart(chart, use_container_width=True)
        else:
            with col3:
                st.plotly_chart(chart, use_container_width=True)
        num += 1
    return client_id
