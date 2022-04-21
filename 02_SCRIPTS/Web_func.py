"""
The purpose of this python file is to regroup every functions for the website
"""
import plotly.graph_objects as go


def value_from_colline(line_id, col, df):
    """
    function to return value of specific column and line using id
    """
    return df[df.SK_ID_CURR.values == line_id].reset_index(drop=True).loc[0, col]


def create_chart(glob_df, grant_df, color_list, column_name):
    """
    Create base of chart with the 2 Dataframes, the list of colors and the name of the feature

    color_list :
        min glob_df - min grant_df
        grant_df min - 25%
        grant_df 25% - 50%
        grant_df 50% - 75%
        grant_df 75% - max
        max grant_df - max glob_df
    """
    first_df_all = glob_df[column_name]
    first_df_grant = grant_df[column_name]
    chart_out = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        gauge={
            'axis': {'range': [first_df_all['min'], first_df_all['max']]},
            'bar': {'color': 'blue'},
            'steps': [
                {'range': [first_df_all['min'], first_df_grant['min']], 'color': color_list[0]},
                {'range': [first_df_grant['min'], first_df_grant['25%']], 'color': color_list[1]},
                {'range': [first_df_grant['25%'], first_df_grant['50%']], 'color': color_list[2]},
                {'range': [first_df_grant['50%'], first_df_grant['75%']], 'color': color_list[3]},
                {'range': [first_df_grant['75%'], first_df_grant['max']], 'color': color_list[4]},
                {'range': [first_df_grant['max'], first_df_all['max']], 'color': color_list[5]}]
        },
        domain={'x': [0, 1], 'y': [0.4, 1]},
        title={'text': column_name}
    ))
    return chart_out
