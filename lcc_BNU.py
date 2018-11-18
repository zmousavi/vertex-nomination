import networkx as nx


#jonathan's code
#Find largest connected component of graph, given nx graph - G


def extract_lcc(G):

	if nx.is_directed(G) == False:
		connComps = max(nx.connected_components(G), key=len)
	else: 
		connComps = max(nx.strongly_connected_components(G), key=len)

	largestComp = G.subgraph(connComps).copy()

	return (largestComp)