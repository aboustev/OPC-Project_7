"""
The purpose of this python file is to regroup every functions from the external kernel
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def semi_detailed_page_front(df_all, client_info, client_id=None):
    cols = [col for col in df_all.columns if col not in ['SK_ID_CURR', 'TARGET', 'target', 'TARGET_PROBA']]
    with st.sidebar:
        select1 = st.selectbox('X_axis', cols)
        select2 = st.selectbox('Y_axis', cols)
        select3 = st.selectbox('Display type', ['Density contour', 'Scatter'])
    client_info['target'] = client_info.TARGET.apply(lambda x: "yes" if x == 1 else "no")
    df_all['target'] = df_all.TARGET.apply(lambda x: "yes" if x == 1 else "no")
    df = df_all[df_all['SK_ID_CURR'] != client_id]

    if select3 == 'Density contour':
        fig = px.density_contour(
            df,
            x=select1,
            y=select2,
            color="target",
            color_discrete_map=dict(yes='green', no='red')
        )
    else:
        fig = px.scatter(
            df,
            x=select1,
            y=select2,
            color="target",
            color_discrete_map=dict(yes='green', no='red'),
            size_max=5
        )
    fig.add_trace(go.Scatter(
        name='Client {}'.format(client_id),
        x=client_info[select1],
        y=client_info[select2],
        mode='markers',
        marker=dict(color='green' if client_info.target[0] == 'yes' else 'red')
    ))
    st.plotly_chart(fig)
