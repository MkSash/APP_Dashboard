import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objs as go
from heapq import nlargest
import warnings
import plotly.figure_factory as ff

def blank_figure():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig


def all_combinations(nodes, edges, categories=['Influencers', 'Followers', 'Bridges', 'Neutrals', 'None']):
    countries = np.concatenate((np.array(['All']), nodes["Country"].unique()), axis=None)
    blank_fig = blank_figure()

    combs = dict()
    for cont in countries:
        if cont == 'All':
            proffessions = np.concatenate((np.array(["All"]), nodes["Profession"].unique()), axis=None)
        else:
            proffessions = np.concatenate((np.array(["All"]), nodes[nodes["Country"] == cont]["Profession"].unique()), axis=None)

        combs[cont] = dict()
        for prof in proffessions:
            combs[cont][prof] = dict()
            if cont == "All" and prof == "All":
                new_nodes = nodes
                new_edges = edges
                G = nx.from_pandas_edgelist(edges, "User_1", "User_2", create_using = nx.Graph())
            
            elif cont == 'All' and prof != 'All':
                new_nodes = nodes[nodes['Profession'] == prof]
                ids = new_nodes["User_ID"]
                new_edges = edges.loc[edges['User_1'].isin(ids)].loc[edges['User_2'].isin(ids)]
                G = nx.from_pandas_edgelist(new_edges, "User_1", "User_2", create_using = nx.Graph())
                
            elif cont != 'All' and prof == 'All':
                new_nodes = nodes[nodes['Country'] == cont]
                ids = new_nodes["User_ID"]
                new_edges = edges.loc[edges['User_1'].isin(ids)].loc[edges['User_2'].isin(ids)]
                G = nx.from_pandas_edgelist(new_edges, "User_1", "User_2", create_using = nx.Graph())
        
            else:
                new_nodes = nodes[(nodes['Country'] == cont) & (nodes['Profession'] == prof)]
                ids = new_nodes["User_ID"]
                new_edges = edges.loc[edges['User_1'].isin(ids)].loc[edges['User_2'].isin(ids)]
                G = nx.from_pandas_edgelist(new_edges, "User_1", "User_2", create_using = nx.Graph())


            combs[cont][prof]['nodes'] = new_nodes
            combs[cont][prof]['edges'] = new_edges

            start_date = new_nodes['Date'].min()
            end_date = new_nodes['Date'].max()
            combs[cont][prof]['start_date'] = start_date
            combs[cont][prof]['end_date'] = end_date

            combs[cont][prof]["G"] = G
            combs[cont][prof]['num_of_nodes'] = G.number_of_nodes()
            combs[cont][prof]['num_of_connections'] = G.number_of_edges()
            if nx.is_empty(G):
                combs[cont][prof]['avg_degree'] = 0
                combs[cont][prof]['bet_fig'] = blank_fig
                combs[cont][prof]['clos_fig'] = blank_fig
                combs[cont][prof]['deg_fig'] = blank_fig
                combs[cont][prof]['eig_fig'] = blank_fig
                combs[cont][prof]['fig_infl'] = blank_fig
                combs[cont][prof]['fig_foll'] = blank_fig
                combs[cont][prof]['fig_brid'] = blank_fig
                combs[cont][prof]['fig_neut'] = blank_fig
                combs[cont][prof]['fig_none'] = blank_fig
            else:
                combs[cont][prof]['avg_degree'] = round(np.mean(list(dict(G.degree).values())), 2)
                combs[cont][prof]['bet_fig'] = betweenness_dist_plot(G)
                combs[cont][prof]['clos_fig'] = closeness_dist_plot(G)
                combs[cont][prof]['deg_fig'] = degree_dist_plot(G)
                combs[cont][prof]['eig_fig'] = eigenvector_dist_plot(G)
                for cat in categories:
                    nodes_to_color = return_nodes(G, cat)
                    combs[cont][prof]['fig_' + cat.lower()[0:4]] = plot_graph(G, nodes_to_color, new_nodes)

    return combs


def return_graph(nodes, edges, start_date, end_date, categories=['Influencers', 'Followers', 'Bridges', 'Neutrals', 'None']):
    blank_fig = blank_figure()
    nodes = nodes[(start_date <= nodes['Date']) & (nodes['Date'] <= end_date)]
    ids = nodes["User_ID"]
    edges = edges.loc[edges['User_1'].isin(ids)].loc[edges['User_2'].isin(ids)]
    G = nx.from_pandas_edgelist(edges, "User_1", "User_2", create_using = nx.Graph())
    combs = dict()
    combs['num_of_nodes'] = G.number_of_nodes()
    combs['num_of_connections'] = G.number_of_edges()
    if nx.is_empty(G):
        combs['avg_degree'] = 0
        combs['bet_fig'] = blank_fig
        combs['clos_fig'] = blank_fig
        combs['deg_fig'] = blank_fig
        combs['eig_fig'] = blank_fig
        for cat in categories:
            combs['fig_' + cat.lower()[0:4]] = blank_fig
        
    else:
        combs['avg_degree'] = round(np.mean(list(dict(G.degree).values())), 2)
        combs['bet_fig'] = betweenness_dist_plot(G)
        combs['clos_fig'] = closeness_dist_plot(G)
        combs['deg_fig'] = degree_dist_plot(G)
        combs['eig_fig'] = eigenvector_dist_plot(G)
        for cat in categories:
            nodes_to_color = return_nodes(G, cat)
            combs['fig_' + cat.lower()[0:4]] = plot_graph(G, nodes_to_color, nodes)
    
    return combs



# function for calculating top N degree/closeness/betweenness/eigenvector centralities
def top_N(graph, n, degree=False, close=False, between=False):
    if degree == True:
        word = "Degree"
        measures = nx.degree_centrality(graph)
        
    elif close == True:
        word = "Closeness"
        measures = nx.closeness_centrality(graph)
        
    elif between == True:
        word = "Betweenness"
        measures = nx.betweenness_centrality(graph)
    
    # taking top N nodes
    top_nodes = nlargest(n, measures, key=measures.get)
    
    # returning list of nodes
    return top_nodes



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
    G_bet = top_N(G, n, between=True)
    G_close = top_N(G, n, close=True)
    
    # list of influencers
    G_infl = list(set(G_bet + G_close))
    
    return G_infl
    


def find_followers(G):    
     # calculating the number of nodes in our data
    nodes = G.number_of_nodes()
    # changing the number of top nodes depending on the number of nodes
    if nodes <= 50:
        n = 7
        m = 5
    elif 50 < nodes <= 100:
        n = 23
        m = 20
    elif 100 < nodes <= 1000:
        n = 55
        m = 50
    else:
        n = 107
        m = 100
    
    # calculating top n nodes with highest betwennes and closeness centrality 
    top_bet = top_N(G, n, between=True)
    top_close = top_N(G, n, close=True)
    
    G_bet = top_bet[-m:]
    G_close = top_close[-m:]

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
    G_bet = top_N(G, n, between=True)
    
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
        line=dict(width=0.5,color='white'),
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
        mode='markers',
        hovertext=[],
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            color=[],
            size=15,
            line=dict(width=0)))

    G_nodes = G.nodes()
    for node in G_nodes:
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
    # coloring and adding informations to nodes
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple(["red" if node in coloring_nodes else "#00A0DC" for node in G_nodes])
        hovertext = str(adjacencies[0]) + ': ' + nodes_left[nodes_left['User_ID'] == adjacencies[0]]['User'].values[0] + "\n" + '# of connections: ' + str(len(adjacencies[1]))
        node_trace['hovertext'] += tuple([hovertext])
        
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
    try:
        fig = ff.create_distplot([betweenness],
                                group_labels = ['Betweenness'],
                                colors=['blue'],
                                show_hist=False,
                                )
        fig.update_layout(showlegend=False,
                        paper_bgcolor='rgba(230,230,250,0.8)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title = 'Betweenness Centrality')
    except np.linalg.LinAlgError:
        fig = {}

    return fig



def closeness_dist_plot(G):
    warnings.filterwarnings('ignore')
    closeness_dict = nx.closeness_centrality(G)
    closeness = list(closeness_dict.values())
    try:
        fig = ff.create_distplot([closeness],
                                group_labels = ['Closeness Centrality'],
                                colors=['blue'],
                                show_hist=False)
        fig.update_layout(showlegend=False,
                        paper_bgcolor='rgba(230,230,250,0.8)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title = 'Closeness Centrality')
    except np.linalg.LinAlgError:
        fig = {}

    return fig



def degree_dist_plot(G):
    warnings.filterwarnings('ignore')
    degree_dict = nx.degree_centrality(G)
    degree = list(degree_dict.values())
    try:
        fig = ff.create_distplot([degree],
                                group_labels = ['Degree Centrality'],
                                colors=['blue'],
                                show_hist=False)
        fig.update_layout(showlegend=False,
                        paper_bgcolor='rgba(230,230,250,0.8)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title = 'Degree Centrality')
    except np.linalg.LinAlgError:
        fig = {}
    return fig

    

    
def eigenvector_dist_plot(G):
    warnings.filterwarnings('ignore')
    eigenvector_dict = nx.eigenvector_centrality(G, max_iter=900)
    eigenvector = list(eigenvector_dict.values())
    try:
        fig = ff.create_distplot([eigenvector],
                                group_labels = ['Eigenvector Centrality'],
                                colors=['blue'],
                                show_hist=False)
        fig.update_layout(showlegend=False,
                        paper_bgcolor='rgba(230,230,250,0.8)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title = 'Eigenvector Centrality')
    except np.linalg.LinAlgError:
        fig = {}
    return fig