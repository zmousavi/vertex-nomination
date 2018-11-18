#!/usr/bin/env python
 # ptr.py
# Created by Hayden Helm, Joshua Agterberg in June 2018. Adapted from Youngser Park.
# Modified by Zeinab
import numpy as np
import networkx as nx
from scipy.stats import rankdata
 #TODO - added a default parameter for the weight column
def pass_to_ranks(G, nedges = 0, weightcol='weight'):
    """
    Passes an adjacency matrix to ranks.
     Inputs
        G - A networkx graph 
    Outputs
        PTR(G) - The passed to ranks version of the adjacency matrix of G
    """
    
    if type(G) == nx.classes.graph.Graph:
        nedges = len(G.edges)
        edges= np.zeros(nedges) #declare float array 
        #loop over the edges and store in an array
         if nx.is_weighted(G, weight=weightcol) == False:
            raise IOError('Weight column not found.')
        
        else:
            j = 0
            for source, target, data in G.edges(data=True):
                edges[j] = data[weightcol]
                j += 1
            
            ranked_values = rankdata(edges)
            #loop through the edges and assign the new weight:
            j = 0
            for source, target, data in G.edges(data=True):
				#This is meant to scale edge weights between 0 and 2
                data[weightcol] = ranked_values[j]*2/(nedges + 1)
                j += 1				
 
        return G