import networkx as nx
import community
from community import community_louvain
import numpy as np
import copy  
import matplotlib.pyplot as plt
import collections

file_108 = r'/Users/z/Desktop/execs_email_t108.txt'

file_109 = r'/Users/z/Desktop/execs_email_t109.txt'


filename = file_108
G_raw = nx.read_weighted_edgelist(filename, comments='#', delimiter=' ', create_using=None, nodetype=str,  encoding='utf-8')
print(G_raw.size())
print(G_raw.number_of_nodes())


filename = file_109
G_raw = nx.read_weighted_edgelist(filename, comments='#', delimiter=' ', create_using=None, nodetype=str,  encoding='utf-8')
print(G_raw.size())
print(G_raw.number_of_nodes())