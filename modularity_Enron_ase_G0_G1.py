#!/usr/bin/env python
# coding: utf-8

# In[24]:


import networkx as nx
import community
from community import community_louvain
import numpy as np
import copy  
import matplotlib.pyplot as plt
import collections
import graph_ase_BNU
from sklearn.utils.extmath import randomized_svd
import collections
import lcc_BNU


# In[17]:


def ase_graph(G): 
    sorted_vertex = sorted(G.nodes())
    A = nx.to_scipy_sparse_matrix(G, nodelist=sorted_vertex)
    n = G.number_of_nodes()
    max_dim = max_dim_ase
    
    svd_seed = 1234
    U, Sigma, VT = randomized_svd(A, 
                              n_components=min(max_dim, n - 1),
                              n_iter=50,
                              random_state=svd_seed)

    ##print ("dimension reduction (elbow selection)")
    #rank_graph =  getElbows_BNU.getElbows(Sigma, n_elbows=elb)
    
    #reduced_dim = rank_graph[(elb-1)]
    ##print ("elbow is %d" %reduced_dim)

    reduced_dim = 2 
    s_sqrt = np.sqrt(Sigma) 

    
    s_sqrt_dim_reduced = s_sqrt[:reduced_dim]
    U_dim_reduced = U[:, :reduced_dim ]
    VT_dim_reduced =VT[:reduced_dim, :]

    Xhat = np.multiply( s_sqrt_dim_reduced, U_dim_reduced)
    
    embedded_dict = {}
    for _, vertex in enumerate(sorted_vertex):
        embedded_dict[vertex] = Xhat[_, :]
     
    
    embedded = collections.namedtuple('embedded', 'Xhat vertex_labels dict')
    result = embedded(Xhat = Xhat, vertex_labels = sorted_vertex, dict = embedded_dict)
    
    return result

def sqdist(vector):
    return sum(x*x for x in vector)

def degtrim_max(G, max_threshold):
    tmpG = copy.deepcopy(G)

    for vertex in G.nodes():
        if G.degree[vertex] >= max_threshold:
            tmpG.remove_node(vertex)

    Graph = copy.deepcopy(tmpG)
    louvain = community.best_partition(Graph, resolution=1, randomize=False)
    mod = community.modularity(louvain, Graph)

    embedded = collections.namedtuple('embedded', 'G mod')
    result = embedded(G = Graph, mod = mod)

    return result

def degtrim_minmax(G, percent_threshold):

    tmpG = copy.deepcopy(G)
    sorted_degrees_desc = sorted(G.degree, key=lambda x: x[1], reverse=True)
    sorted_degrees_asc = sorted(G.degree, key=lambda x: x[1], reverse=False)
    count_threshold = int(percent_threshold * G.number_of_nodes())
    ctr = 0 
    if ctr <= count_threshold:
        high_trim = sorted_degrees_desc[ctr][0]
        low_trim = sorted_degrees_asc[ctr][0]
        tmpG.remove_node(high_trim)
        tmpG.remove_node(low_trim)
        ctr += 1
        

    Graph = copy.deepcopy(tmpG)
    louvain = community.best_partition(Graph, resolution=1, randomize=False)
    mod = community.modularity(louvain, Graph)

    embedded = collections.namedtuple('embedded', 'G mod')
    result = embedded(G = Graph, mod = mod)

    return result

#procrustes
def procrustes(A, B):
    tmp = A.T @ B
    U_tmp, Sigma_tmp, VT_tmp = np.linalg.svd(tmp)
    W = U_tmp @ VT_tmp
    return W
    #A@W - B


# In[48]:


file_G0 = r'/Users/z/Desktop/execs_email_tG108.txt'
file_G1 = r'/Users/z/Desktop/execs_email_tG109.txt'
#file_G1 = r'/Users/z/Desktop/execs_email_tG107.txt'


# In[50]:


G0 = nx.read_weighted_edgelist(file_G0, comments='#', delimiter=' ', create_using=None, nodetype=str,  encoding='utf-8')
G1 = nx.read_weighted_edgelist(file_G1, comments='#', delimiter=' ', create_using=None, nodetype=str,  encoding='utf-8')




print(G0.size())
print(G0.number_of_nodes())
print(G1.size())
print(G1.number_of_nodes())


# In[52]:


G1_misses = set(G0.nodes()) - set(G1.nodes())
G0_misses = set(G1.nodes()) - set(G0.nodes())


# In[11]:


# G0_world =  G_G0
# G0_world.add_nodes_from(G_G1)
# G1_world =  G_G1
# G1_world.add_nodes_from(G_G0)


# In[119]:


# print(G0.size())
# print(G0.number_of_nodes())
# print(G1.size())
# print(G1.number_of_nodes())
# print(G0_world.size())
# print(G0_world.number_of_nodes())
# print(G1_world.size())
# print(G1_world.number_of_nodes())


# In[53]:


Graph = copy.deepcopy(G_G0)
louvain_G0 = community.best_partition(Graph, resolution=1, randomize=False)
mod_G0 = community.modularity(louvain_G0, Graph)
print(mod_G0)

Graph = copy.deepcopy(G_G1)
louvain_G1 = community.best_partition(Graph, resolution=1, randomize=False)
mod_G1 = community.modularity(louvain_G1, Graph)
print(mod_G1)


# In[54]:


G = G_G0
tot_degree_sequence_G0 = [d for n, d in G.degree()]  # degree sequence

G = G_G1
tot_degree_sequence_G1 = [d for n, d in G.degree()]  # degree sequence


# In[55]:


fig = plt.figure()
#fig.tight_layout()
plt.figure(figsize=(6, 4), dpi=80)

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 1   # the amount of width reserved for blank space between subplots
hspace = 1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)



plt.subplot(1, 2, 1)
plt.hist(tot_degree_sequence_G0, width=0.80, color='b')
plt.title("Tot-degree G_G0")
plt.ylabel("Count")
#plt.xlabel("Out Degree")
plt.ylim(0, 60)

plt.subplot(1, 2, 2)
plt.hist(tot_degree_sequence_G1, width=0.80, color='b')
plt.title("Tot-degree G_G1")
plt.ylabel("Count")
#plt.xlabel("In Degree")
#plt.ylim(0, 60)


# In[56]:


G = G0

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots(figsize=(18,10))
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

# draw graph in inset
plt.axes([0.4, 0.4, 0.5, 0.5])
Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = nx.spring_layout(G)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.show()

deg_G0 = deg


# In[67]:


G = G1

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots(figsize=(18,10))
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

# draw graph in inset
plt.axes([0.4, 0.4, 0.5, 0.5])
Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = nx.spring_layout(G)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.show()


# In[68]:


graphs_modularity3 = []

#percent_threshold = [0.01, 0.02, 0.05, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4] #this was for G0

percent_threshold = [0.01, 0.02, 0.05, 0.08] #, 0.1, 0.13, 0.16] #, 0.17, 0.2, 0.25, 0.3] #, 0.35]#, 0.4] 
                     
G_trimmed_new = G1
for percent in percent_threshold :
    Gtrim_mod = degtrim_minmax(G_trimmed_new, percent)
    G_trimmed_new = Gtrim_mod.G
    graphs_modularity3.append(Gtrim_mod.mod) 
Gtrim = G_trimmed_new

print (graphs_modularity3)

#print(G_trim.nodes())

degree_sequence = sorted([d for n, d in Gtrim.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram after breakthrough")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)


# In[69]:


G = lcc_BNU.extract_lcc(G1)

fig, ax = plt.subplots(figsize=(30,10))

plt.title("Gnoise_lcc")

plt.subplot(1, 3, 1)

Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
pos = nx.spring_layout(G)
plt.axis('off')
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.subplot(1, 3, 2)
G2 = lcc_BNU.extract_lcc(G0)

Gcc = sorted(nx.connected_component_subgraphs(G2), key=len, reverse=True)[0]
pos = nx.spring_layout(G2)
plt.axis('off')
nx.draw_networkx_nodes(G2, pos, node_size=20, node_color='blue')
nx.draw_networkx_edges(G2, pos, alpha=0.4)
nx.draw_networkx_labels(G2, pos)

plt.subplot(1, 3, 3)

G3 = lcc_BNU.extract_lcc(Gtrim)
Gcc = sorted(nx.connected_component_subgraphs(G3), key=len, reverse=True)[0]
pos = nx.spring_layout(G3)
plt.axis('off')
nx.draw_networkx_nodes(G3, pos, node_size=20, node_color = 'green',  with_labels = True)
nx.draw_networkx_edges(G3, pos, alpha=0.4)
nx.draw_networkx_labels(G3, pos)

plt.show()


# In[60]:


max_dim_ase = 2

ase_G0 = ase_graph(G0)
ase_G1 = ase_graph(G1)
ase_Gtrim = ase_graph(Gtrim)


# In[61]:


print(set(G1.nodes()) - set(Gtrim.nodes()))
#note here trimming only trimmed nodes that were common in both G0 and G1
#remember to do procrustes!!!!! 

intersecting_G0_noise =   set(G0.nodes()).intersection(set(G1.nodes()))
intersecting_all3 =   list(set(intersecting_G0_noise).intersection(set(Gtrim.nodes())))

print(len(intersecting_G0_noise))
print(len(intersecting_all3))


# In[62]:



intersecting_index = []
for v in intersecting_all3:
    intersecting_index.append(ase_G0.vertex_labels.index(v))
XhatG0_shared = ase_G0.Xhat[intersecting_index]

intersecting_index = []
for v in intersecting_all3:
    intersecting_index.append(ase_G1.vertex_labels.index(v))
XhatG1_shared = ase_Gnoise.Xhat[intersecting_index]

intersecting_index = []
for v in intersecting_all3:
    intersecting_index.append(ase_Gtrim.vertex_labels.index(v))
XhatG1_Trimmed_shared = ase_Gtrim.Xhat[intersecting_index]


# In[63]:


#procrustes
W_G1_G0 = procrustes(XhatNoise_shared, XhatG0_shared)
XhatG1_shared_proj = XhatG1_shared@W_G1_G0 


W_G1_Trimmed_G0 = procrustes(XhatG1_Trimmed_shared, XhatG0_shared)
XhatG1_Trimmed_shared_proj = XhatG1_Trimmed_shared@W_G1_Trimmed_G0 


# In[64]:


Xhat_G0_G1 = XhatG0_shared - XhatG1_shared_proj 
diff_G0_G1 = np.linalg.norm(Xhat_G0_G1, axis=1)


# In[65]:


Xhat_G0_G1 = XhatG0_shared - XhatG1_shared_proj 


plt.plot(XhatG0_shared[:,0], XhatG0_shared[:,1], 'bo')
plt.plot(XhatG1_shared_proj [:,0], XhatG1_shared_proj [:,1], 'ro')
plt.plot(XhatG1_Trimmed_shared_proj [:,0], XhatG1_Trimmed_shared_proj [:,1], 'go')


# In[66]:


Xhat_G0_G1_Trimmed = XhatG0_shared - XhatG1_Trimmed_shared_proj 
diff_G0_G1_Trimmed = np.linalg.norm(Xhat_G0_G1_Trimmed, axis=1)
#diff_G0_G1_Trimmed
a = np.array(diff_G0_G1)
b = np.array(diff_G0_G1_Trimmed)
m = np.vstack((a,b))
#m1 = np.asmatrix(m)
plt.figure(figsize=(10, 10), dpi=80)
plt.imshow(m, cmap='hot', interpolation='nearest')
plt.show()
#black means small number


# In[46]:


rank_trim = []
rank_noise = []

idx = 1 

for idx in range(XhatG0_shared.shape[0]):
    distance_to_noisy = XhatG1_shared_proj - XhatG0_shared[idx,]
    diff_G1_G0v = np.linalg.norm(distance_to_noisy, axis=1)
    distance_to_trim = XhatG1_Trimmed_shared_proj - XhatG0_shared[idx,]
    diff_G1_Trimmed_G0v = np.linalg.norm(distance_to_trim, axis=1)
    rank_trim.append(diff_G1_Trimmed_G0v.argsort()[0]) 
    rank_noise.append(diff_G1_G0v.argsort()[0])


delta_rank = np.array(rank_noise) - np.array(rank_trim)


# In[47]:


plt.plot(np.array(rank_noise), 'ro')
plt.plot(np.array(rank_trim), 'go')


# In[292]:
