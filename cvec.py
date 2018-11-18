# Created by Zeinab Mousavi in August 2018. 
import numpy as np
import networkx as nx
def diag_aug(G, weightcol='weight'):
    """
    Performs diagonal augmentation on G.
     Inputs
        G - A networkx graph 
    Outputs
        aug(G) - The diagonal augmented version of the adjacency matrix of G
    """
    if type(G) == nx.classes.graph.Graph:
        vcount = len(G.nodes())
        degrees = np.zeros(vcount) #declare float array
        for vertex in G.nodes():
            G.add_edge(vertex, vertex)
            G[vertex][vertex][weightcol] = G.degree(vertex)/(vcount - 1)
         
    return G