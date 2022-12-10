import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import datetime as dt
import networkx as nx

from functions import *




edges = pd.read_csv('./data/edges.csv', usecols = ["User_1", "User_2"])
nodes = pd.read_csv('./data/nodes.csv', usecols = ["User_ID", "User", "Country", "Profession", "Date"])
#dropping duplicate rows
edges.drop_duplicates(inplace=True)
nodes.drop_duplicates(inplace=True)

nodes['Date'] = pd.to_datetime(nodes['Date']).dt.date

all_combs = all_combinations(nodes, edges)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#080844',
    'text': '#EAEAFF'
}

app.layout = html.Div(style={'padding': 20}, className="container", children=[
    html.Div(className="row", children=[
        html.H1('Social Network Visualizer', className = 'nine columns',style={'fontSize': 50,
                                                    'color': colors['text'], 'fontFamily': "Serif"}),
    html.Div(id = "btn",className = "three columns",children=[
    html.Button("Download the Data",style = { 'border-color': 'rgb(194, 221, 244)' }, id="download-btn", n_clicks=0),
    dcc.Download(id="download")
]),
        html.Div([
            html.P( 'Visualize your Linkedin Connections Network', style={
                   'color': colors['text'], 'fontSize': 20, 'fontFamily': "Serif"},className = "eight columns")
        ])
    ]),
     html.Div( className='row', id ='box-container', children=[
        
        dbc.Card(className='three columns', id='first-box',children=[dbc.CardBody(
        [
            html.H4("Number of Nodes", className="card-title"),
            html.P(id="display-nodes",className="card-text"),
        ]
    )],
        style={"width": "18rem"},
    ),

         dbc.Card(className='three columns',id='second-box', children=[dbc.CardBody(
        [
            html.H4("Number of Connections", className="card-title"),
            html.P(id="display-con",className="card-text"),
        ]
    )],
        style={"width": "18rem"},
    ) 
    ,

         dbc.Card(className='three columns',id='third-box', children=[dbc.CardBody(
        [
            html.H4("Average Degree", className="card-title"),
            html.P(id="display-deg", className="card-text"),
        ]
    )],
        style={"width": "18rem"},
    )
])

    ,
      html.Br(),
    html.Div(className="row", children=[
        html.Div(className='eight columns', children=[
            dcc.Graph(id='sna_graph', style={'padding': 20}, figure = blank_figure())
        ]),
        html.Div(className="four columns", children=[

            html.Div([
                html.P('Filter the Graph', style={
                       'color': colors['text'], 'fontSize': 20, 'fontFamily': "Serif", 'margin-top': '20px'}),
            ]),

            html.Div(style={'padding': 10, 'flex': 1}, children=[
                html.Label('Select the Country ', style={
                           'color': colors['text'], 'fontFamily': "Serif"}),
                dcc.Dropdown(
                    options=np.concatenate(
                        (np.array(['All']), nodes["Country"].unique()), axis=None),
                    value='All',
                    id='select-country'
                ),

                html.Br(),

                dcc.Dropdown(
                    id='select-profession'
                ),

                html.Br(),

                html.Label('Highlight: ', style={
                           'color': colors['text'], 'fontFamily': "Serif"}),
                dcc.RadioItems(options = ['Influencers', 'Followers', 'Bridges', 'Neutrals',
                               'None'], value= 'None', id = 'node-type', style={'color': colors['text']}),

                html.Br(),
##################################################################################################################
                html.Div([
                    dcc.DatePickerRange(
                        id = 'date-picker',
                        min_date_allowed = pd.to_datetime(nodes['Date']).dt.date.min(),
                        max_date_allowed = pd.to_datetime(nodes['Date']).dt.date.max(),
                        display_format = 'MMM YYYY'
                    )
                ])
                

            ]),

        ]),
########################################################################################
        html.Div(className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Graph(id='bet_plot', style={'padding': 20}, figure = blank_figure()),
            ]),

            html.Div(className='six columns', children=[
                dcc.Graph(id='close_plot', style={'padding': 20}, figure = blank_figure())
            ])
        ]),

        html.Div(className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Graph(id='deg_plot', style={'padding': 20}, figure = blank_figure())
            ]),

            html.Div(className='six columns', children=[
                dcc.Graph(id='eig_plot', style={'padding': 20}, figure = blank_figure())
            ])
        ])
###########################################################################################
    ])

])

# selectiong options for proffessions based on country
@app.callback(
    Output('select-profession', 'options'),
    Input('select-country', 'value')
)
def select_profession(country):
    if country == "All":
        proffessions = np.concatenate((np.array(['All']), nodes['Profession'].unique()), axis=None)
    else:
        proffessions = np.concatenate((np.array(['All']), nodes[nodes['Country'] == country]['Profession'].unique()), axis=None)
    return proffessions


# passing options for proffessions to values
@app.callback(
    Output('select-profession', 'value'),
    Input('select-profession', 'options')
)
def show_professions(options):
    return options[0]


# changing date based on country and profession
@app.callback(
    Output('date-picker', 'start_date'),
    Output('date-picker', 'end_date'),
    Input('select-country', 'value'),
    Input('select-profession', 'value')
)
def change_date(country, profession):
    start_date = all_combs[country][profession]['start_date']
    end_date = all_combs[country][profession]['end_date']
    return start_date, end_date



@app.callback(
    Output('display-nodes', 'children'),
    Output('display-con', 'children'),
    Output('display-deg', 'children'),
    Output('sna_graph', 'figure'),
    Output('bet_plot', 'figure'),
    Output('close_plot', 'figure'),
    Output('deg_plot', 'figure'),
    Output('eig_plot', 'figure'),
    Input('select-country', 'value'),
    Input('select-profession', 'value'),
    Input('node-type', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date'),
    )
def sna_graph(country, profession, button, start_date, end_date):
    
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date() 
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()

    if ((start_date == all_combs[country][profession]['start_date']) & (end_date == all_combs[country][profession]['end_date'])):
        num_of_nodes = all_combs[country][profession]['num_of_nodes']
        num_of_connections = all_combs[country][profession]['num_of_connections']
        avg_degree = all_combs[country][profession]['avg_degree']
        fig = all_combs[country][profession]['fig_' + button.lower()[0:4]]
        bet_fig = all_combs[country][profession]['bet_fig']
        clos_fig = all_combs[country][profession]['clos_fig']
        deg_fig = all_combs[country][profession]['deg_fig']
        eig_fig = all_combs[country][profession]['eig_fig']
    else:
        new_nodes = all_combs[country][profession]['nodes']
        new_edges = all_combs[country][profession]['edges']
        combs = other_combinations(new_nodes, new_edges, start_date, end_date)
        num_of_nodes = combs['num_of_nodes']
        num_of_connections = combs['num_of_connections']
        avg_degree = combs['avg_degree']
        fig = combs['fig_' + button.lower()[0:4]]
        bet_fig = combs['bet_fig']
        clos_fig = combs['clos_fig']
        deg_fig = combs['deg_fig']
        eig_fig = combs['eig_fig']

    return num_of_nodes, num_of_connections, avg_degree, fig, bet_fig, clos_fig, deg_fig, eig_fig#, dcc.send_data_frame(nodes_left.to_csv, filename="nodes.csv")

@app.callback(
    Output("download", "data"),
    Input("download-btn", "n_clicks"),
    [State('select-country','value'),
    State('date-picker', 'start_date'),
    State('date-picker', 'end_date')],
    prevent_initial_call=True,
)
def func(n_clicks, country, profession, start_date, end_date):
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date() 
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d').date()

    nodes = all_combs[country][profession]
    nodes = nodes[(start_date <= nodes['Date']) & (nodes['Date'] <= end_date)]
  
    return dcc.send_data_frame(nodes.to_csv, "mydf.csv")


if __name__ == '__main__':
    app.run_server(debug=True)
