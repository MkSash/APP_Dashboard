import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objs as go
from heapq import nlargest, nsmallest
import warnings
import plotly.figure_factory as ff


# function for calculating top N degree/closeness/betweenness/eigenvector centralities
def top_nottop_N(graph, n, top=False, not_top=False, degree=False, close=False, between=False, eigen=False):
    if degree == True:
        word = "Degree"
        measures = nx.degree_centrality(graph)
        
    elif close == True:
        word = "Closeness"
        measures = nx.closeness_centrality(graph)
        
    elif between == True:
        word = "Betweenness"
        measures = nx.betweenness_centrality(graph)
        
    elif eigen == True:
        word = "Eigenvector"
        # calculating in and out eigenvector centralities
        measures_in = nx.eigenvector_centrality(graph, max_iter=1000)
        measures_out = nx.eigenvector_centrality(graph.reverse(), max_iter=1000)
        
        measures = {}
        # combining them by calculating mean
        for key in measures_in.keys():
            measures[key] = (measures_in[key] + measures_out[key])/2
    
    if top == True:
        # taking top N nodes
        top_nodes = nlargest(n, measures, key=measures.get)

    elif not_top == True:
        # taking not top N nodes
        top_nodes = nsmallest(n, measures, key=measures.get)
    
    # returning list of nodes
    return top_nodes


def return_graph(nodes, edges, country, profession, start_date, end_date):

    nodes = nodes[(start_date <= pd.to_datetime(nodes['Date'])) & (pd.to_datetime(nodes['Date']) <= end_date)]
    if country == 'All' and profession == 'All':
        G = nx.from_pandas_edgelist(edges, "User_1", "User_2", create_using = nx.Graph())
        
    elif country == 'All' and profession != 'All':
        nodes = nodes[nodes['Profession'] == profession]
        prof_nodes = np.array(nodes['User_ID'])
        prof_edges = edges.loc[edges['User_1'].isin(prof_nodes)].loc[edges['User_2'].isin(prof_nodes)]
        G = nx.from_pandas_edgelist(prof_edges, "User_1", "User_2", create_using = nx.Graph())
        
    elif country != 'All' and profession == 'All':
        nodes = nodes[nodes['Country'] == country]
        country_nodes = np.array(nodes['User_ID'])
        country_edges = edges.loc[edges['User_1'].isin(country_nodes)].loc[edges['User_2'].isin(country_nodes)]
        G = nx.from_pandas_edgelist(country_edges, "User_1", "User_2", create_using = nx.Graph())
        
    else:
        nodes = nodes[(nodes['Country'] == country) & (nodes['Profession'] == profession)]
        the_nodes = np.array(nodes['User_ID'])
        the_edges = edges.loc[edges['User_1'].isin(the_nodes)].loc[edges['User_2'].isin(the_nodes)]
        G = nx.from_pandas_edgelist(the_edges, "User_1", "User_2", create_using = nx.Graph())

        
    return G, nodes



def find_influancers(G):
    # calculating the number of nodes in our data
    nodes = G.number_of_nodes()
    # changing the number of top nodes depending on the number of nodes
    if nodes <= 50:
        n = 2
    elif 50 < nodes <= 100:
        n = 3
    elif 100 < nodes <= 1000:
        n = 5
    else:
        n = 7
    
    # calculating top n nodes with highest betwennes and closeness centrality 
    G_bet = top_nottop_N(G, n, top=True, between=True)
    G_close = top_nottop_N(G, n, top=True, close=True)
    
    # list of influencers
    G_infl = list(set(G_bet + G_close))
    
    return G_infl
    


def find_followers(G):    
     # calculating the number of nodes in our data
    nodes = G.number_of_nodes()
    # changing the number of top nodes depending on the number of nodes
    if nodes <= 50:
        n = 5
    elif 50 < nodes <= 100:
        n = 20
    elif 100 < nodes <= 1000:
        n = 50
    else:
        n = 100
    
    # calculating top n nodes with highest betwennes and closeness centrality 
    G_bet = top_nottop_N(G, n, not_top=True, between=True)
    G_close = top_nottop_N(G, n, not_top=True, close=True)
    
    # list of influencers
    G_foll = list(set(G_bet + G_close))
    
    return G_foll



def find_bridges(G):
    # calculating the number of nodes in our data
    nodes = G.number_of_nodes()
    # changing the number of top nodes depending on the number of nodes
    if nodes <= 50:
        n = 2
    elif 50 < nodes <= 100:
        n = 3
    elif 100 < nodes <= 1000:
        n = 5
    else:
        n = 7
    
    # calculating top n nodes with highest betwennes and closeness centrality 
    G_bet = top_nottop_N(G, n, top=True, between=True)
    
    return G_bet


def find_neutrals(G):
    # finding influencers, followers, and bridges
    G_infl = find_influancers(G)
    G_foll = find_followers(G)
    G_brid = find_bridges(G)
    nodes = set(G.nodes())
    # nodes that need to be removes
    remove = set(G_infl + G_foll + G_brid)
    # removing nodes
    G_neut =  list(nodes - remove)
    
    return G_neut



def return_nodes(G, word):

    if word == 'Influencers':
        nodes = find_influancers(G)

    elif word == 'Followers':
        nodes = find_followers(G)

    elif word == 'Bridges':
        nodes = find_bridges(G)

    elif word == 'Neutrals':
        nodes = find_neutrals(G)

    else:
        nodes = []
    
    return nodes



def plot_graph(G, coloring_nodes, nodes_left):
    
    pos = nx.spring_layout(G, seed=1111)
    
    # adding positions to graph
    for node, position in pos.items():
        G.nodes[node]['pos'] = position
        
    # edges for plot
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        
    # nodes for plot
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hovertext=[],
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    G_nodes = G.nodes()
    for node in G_nodes:
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
    # coloring and adding informations to nodes
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple(["red" if node in coloring_nodes else "#00A0DC" for node in G_nodes])
        hovertext = str(adjacencies[0]) + ': ' + nodes_left[nodes_left['User_ID'] == adjacencies[0]]['User'].values[0]
        text = ' # of connections: ' + str(len(adjacencies[1]))
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
        
    # defining figure
    fig = {"data": [edge_trace, node_trace],
           "layout": go.Layout(
                # title='Connection Plot',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))}
    
    # returning plot
    return fig



def betweenness_dist_plot(G):
    warnings.filterwarnings('ignore')
    betweenness_dict = nx.betweenness_centrality(G) 
    betweenness = list(betweenness_dict.values())
    fig = ff.create_distplot([betweenness],
                             group_labels = ['Betweenness'],
                             colors=['blue'],
                             show_hist=False,
                            )
    fig.update_layout(showlegend=False,
                      paper_bgcolor='rgba(230,230,250,0.8)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title = 'Betweenness Centrality')
    return fig



def closeness_dist_plot(G):
    warnings.filterwarnings('ignore')
    closeness_dict = nx.closeness_centrality(G)
    closeness = list(closeness_dict.values())
    fig = ff.create_distplot([closeness],
                             group_labels = ['Closeness Centrality'],
                             colors=['blue'],
                             show_hist=False)
    fig.update_layout(showlegend=False,
                      paper_bgcolor='rgba(230,230,250,0.8)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title = 'Closeness Centrality')

    return fig



def degree_dist_plot(G):
    warnings.filterwarnings('ignore')
    degree_dict = nx.degree_centrality(G)
    degree = list(degree_dict.values())
    fig = ff.create_distplot([degree],
                             group_labels = ['Degree Centrality'],
                             colors=['blue'],
                             show_hist=False)
    fig.update_layout(showlegend=False,
                      paper_bgcolor='rgba(230,230,250,0.8)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title = 'Degree Centrality')
    return fig
    

    
def eigenvector_dist_plot(G):
    warnings.filterwarnings('ignore')
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=900)
    eigenvector = list(eigenvector_dict.values())
    fig = ff.create_distplot([eigenvector],
                             group_labels = ['Eigenvector Centrality'],
                             colors=['blue'],
                             show_hist=False)
    fig.update_layout(showlegend=False,
                      paper_bgcolor='rgba(230,230,250,0.8)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title = 'Eigenvector Centrality')
    return fig