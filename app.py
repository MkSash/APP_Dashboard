import dash
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as col
from operator import itemgetter
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output
from functions import *
from datetime import date



edges = pd.read_csv('./data/edges.csv', usecols = ["User_1", "User_2"])
nodes = pd.read_csv('./data/nodes.csv', usecols = ["User_ID", "User", "Country", "Profession", "Date"])
#dropping duplicate rows
edges.drop_duplicates(inplace=True)
nodes.drop_duplicates(inplace=True)

# nodes["Month_Year"] = date(nodes["Date"])

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
    html.Button("Download the Data",style = { 'border-color': 'rgb(194, 221, 244)' }, id="download_btn",n_clicks=5),
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
            dcc.Graph(id='sna_graph', style={'padding': 20})
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
                    id='select_country'
                ),
                html.Br(),
                dcc.Dropdown(
                    options=np.concatenate(
                        (np.array(['All']), nodes["Profession"].unique()), axis=None),
                    value='All',
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
                        # min = pd.to_datetime(nodes['Date']).dt.date.min(),
                        # max = pd.to_datetime(nodes['Date']).dt.date.max(),
                        # value = pd.to_datetime(nodes['Date']).dt.date.max(),
                        start_date = pd.to_datetime(nodes['Date']).dt.date.min(),
                        end_date = pd.to_datetime(nodes['Date']).dt.date.max(),
                        min_date_allowed = pd.to_datetime(nodes['Date']).dt.date.min(),
                        max_date_allowed = pd.to_datetime(nodes['Date']).dt.date.max(),
                        display_format = 'MMM YYYY'
                    )
                        # value = nodes["Date"].max())
                ])
                

            ]),

        ]),
########################################################################################
        html.Div(className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Graph(id='bet_plot', style={'padding': 20}),
            ]),

            html.Div(className='six columns', children=[
                dcc.Graph(id='close_plot', style={'padding': 20})
            ])
        ]),

        html.Div(className="row", children=[
            html.Div(className='six columns', children=[
                dcc.Graph(id='deg_plot', style={'padding': 20})
            ]),

            html.Div(className='six columns', children=[
                dcc.Graph(id='eig_plot', style={'padding': 20})
            ])
        ])
###########################################################################################
    ])

])

@app.callback(
    Output('sna_graph', 'figure'),
    Output('display-nodes', 'children'),
    Output('display-con', 'children'),
    Output('display-deg', 'children'),
    Output('bet_plot', 'figure'),
    Output('close_plot', 'figure'),
    Output('deg_plot', 'figure'),
    Output('eig_plot', 'figure'),
    # Output('download', 'data'),
    Input('select_country', 'value'),
    Input('select-profession', 'value'),
    Input('node-type', 'value'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
    # Input('date-slider', 'value'),
    # Input('download_btn', 'n_clicks')
    
    )
def sna_graph(country, profession, button, start_date, end_date):# , n_clicks)
    G, nodes_left = return_graph(nodes, edges, country, profession, start_date, end_date)
    coloring_nodes = return_nodes(G, button)
    fig = plot_graph(G, coloring_nodes, nodes_left)

    num_of_nodes = G.number_of_nodes()
    num_of_connections = G.number_of_edges()
    avg_degrees = round(np.mean(list(dict(G.degree).values())),2)

    bet_fig = betweenness_dist_plot(G)
    close_fig = closeness_dist_plot(G)
    deg_fig = degree_dist_plot(G)
    eig_fig = eigenvector_dist_plot(G)

    return fig, num_of_nodes, num_of_connections,  avg_degrees, bet_fig, close_fig, deg_fig, eig_fig#, dcc.send_data_frame(nodes_left.to_csv, filename="nodes.csv")

if __name__ == '__main__':
    app.run_server(debug=True)
