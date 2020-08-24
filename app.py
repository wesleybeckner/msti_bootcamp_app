# -*- coding: utf-8 -*-
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import json

import numpy as np
import pandas as pd
import fnmatch



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1',
                  '#FF9DA7', '#9C755F', '#BAB0AC']

df = pd.read_csv('data/entry_survey.csv')

ranks = ['1st choice', '2nd choice', '3rd choice', '4th choice']
identity = ['Timestamp', 'Username', 'Preferred Name']
learning = ['Preferred Learning Topics, {}'.format(i) for i in ranks]
application = ['Preferred Application Topics, {}'.format(i) for i in ranks]
language = ['Preferred Language Topics, {}'.format(i) for i in ranks]
python = ['Python Proficiency']
comfort = ['Virtual Learning Comfort Level']
working = ['Years Working']
style = ['Learning Style']
myers = ['Myers-Briggs Personality Type']
quiz = [' * . / - + **, are all examples of what in Python?']
columns = identity + learning + application + language + python + comfort + working + style + myers + quiz
number_columns = python + comfort + working
mult_choice_columns = style + myers + quiz
ranking_columns = ['Learning Topics', 'Application Topics', 'Language Topics']
question = ranking_columns + number_columns + mult_choice_columns + ['Myers-Briggs Letters']
df.columns = columns
df = df[[i for i in df.columns if i not in identity]]

def make_figure(question, df=df):
    fig = go.Figure()
    if question in number_columns:
        sub_df = df[question].value_counts()
        to_plot = pd.DataFrame(sub_df)
        fig = px.bar(to_plot, x=to_plot.index, y=to_plot.columns[0],
                    labels={to_plot.columns[0]: 'Number of Students', 'x': to_plot.columns[0]})
        fig.update_layout(xaxis=dict(
        tickvals = [0, 1, 2, 3, 4, 5],
        ticktext = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']))
    if question in ranking_columns:
        sub_df = df[[i for i in df.columns if question in i]]
        sub_df.columns = [i.split(',')[1] for i in sub_df.columns]
        to_plot = sub_df.apply(pd.Series.value_counts).T
        fig = go.Figure(data=[go.Bar(name=j, x=to_plot.index, y=to_plot[j])
                              for i, j in zip(to_plot.index, to_plot.columns)],
                       )
    elif question in mult_choice_columns:
        sub_df = df[question]
        to_plot = pd.DataFrame(sub_df.value_counts())
        fig = px.bar(to_plot, y=to_plot.index, x=to_plot.columns[0],
                     orientation='h', labels={to_plot.columns[0]: 'Number of Students', 'y': to_plot.columns[0]})
    elif question in ['Myers-Briggs Letters']:
        question = mult_choice_columns[1]
        sub_df = df[question]
        to_plot = pd.DataFrame(sub_df.value_counts())
        letters = [['E', 'I'], ['N', 'S'], ['F', 'T'], ['P', 'J']]
        fig = go.Figure()
        for pair in letters:
            first_value = to_plot.loc[fnmatch.filter(to_plot.index, '*–*{}* *'.format(pair[0]))].sum()
            second_value = to_plot.loc[fnmatch.filter(to_plot.index, '*–*{}* *'.format(pair[1]))].sum()
            fig.add_trace(go.Bar(name=pair[0], x=[pair[0]], y=first_value,  marker_color='orchid', showlegend=False))
            fig.add_trace(go.Bar(name=pair[1], x=[pair[1]], y=second_value, marker_color='coral', showlegend=False))
    fig.update_layout(clickmode='event+select')
    return fig

HIDDEN = html.Div([
    html.Div(id='df',
             style={'display': 'none'},
             children=df.to_json()),
             ])
app.layout = html.Div([
    HIDDEN,
    html.Div([
        dcc.Dropdown(
            id='bar_plot_1_options',
            options=[{'label': i, 'value': i} for i in question],
            value=question[0]
        ),
        ],),
    html.Div([
        dcc.Graph(
            id='bar_plot_1',
            figure = make_figure(question[0])
        )
    ]),
    html.Div(
        children=dash_table.DataTable(id='table',
                            sort_action='native',
                            columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                            data = df.to_dict('rows'),
                            editable=True,
                            filter_action="native",
                            sort_mode="multi",
                            column_selectable="single",
                            row_selectable="multi",
                            row_deletable=True,
                            selected_columns=[],
                            # selected_rows=[0, 1, 2],
                            page_action="native",
                            page_current= 0,
                            style_table={
                                    'maxHeight': '50ex',
                                    'overflowY': 'scroll',
                                    'width': '100%',
                                    'minWidth': '100%',
                                }
                            ),
                            ),
    # html.Pre(id='selected-data'),
], className='mini_container',
   id='dashcontainer',
   style={'display': 'block'},)

# @app.callback(
#     Output('selected-data', 'children'),
#     [Input('bar_plot_1', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData, indent=2)

@app.callback(
    [Output('bar_plot_1', 'figure'),
     Output('bar_plot_1', 'selectedData')],
    [Input('bar_plot_1_options', 'value'),
    Input('table', 'derived_virtual_selected_rows'),
    Input('table', 'derived_virtual_data'),])
def update_download_link(question, rows, data):
    if (data is not None) and (len(rows) == 0):
        return [make_figure(question, pd.DataFrame(data)), None]
    elif (data is not None) and (len(rows) > 0):
        return [make_figure(question, pd.DataFrame(data).iloc[rows]), None]
    else:
        return [make_figure(question), None]



if __name__ == '__main__':
    app.run_server(debug=True)
