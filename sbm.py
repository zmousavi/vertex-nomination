

import numpy as np 
import matplotlib.pyplot as plt
import graph_ase_BNU
import graph_lse_BNU
import networkx as nx
import lcc_BNU 
import getElbows_BNU
from sklearn.utils.extmath import randomized_svd
import gmm_BNU
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import itertools
import pdb
import cvec



K = 4
n = 1000

#B = [[0.9, 0.02, .02, .02], [.02, 0.9, .02, .02], [0.02, 0.02, 0.9, 0.02], [0.02, 0.02, 0.02, 0.9]]
#block_sizes = [int(n/K) for block in range(K)]


B = np.array(
	[[0.0087900779, 0.018577700, 0.0004554752, 0.001987937],
	[0.0185776999, 0.076861773, 0.0022648039, 0.012675506],
	[0.0004554752, 0.002264804, 0.0088418623, 0.018968512],
	[0.0019879368, 0.012675506, 0.0189685119, 0.075949405]])

block_sizes = np.array([280,220, 280,220])

vertex_labels = []
for i in range(len(block_sizes)):
	a = block_sizes[i]*[i]
	vertex_labels.extend(a)


A_raw = np.zeros((n, n), dtype=int)



for row in range(n):
	for col in range((row+1), n):
		source_community = vertex_labels[row]
		target_community = vertex_labels[col]
		p = B[source_community][target_community] 
		s = np.random.binomial(n=1, p=p, size=1) 
		if s >= 0.5:
			A_raw[row][col] = 1
			A_raw[col][row] = 1


plt.imshow(A_raw, cmap='hot_r', interpolation='nearest')
output_fig = '/Users/z/Desktop/test.png'
plt.savefig(output_fig, bbox_inches='tight')


G_raw = nx.from_numpy_matrix(A_raw)



G_lcc = lcc_BNU.extract_lcc(G_raw)
n_lcc = G_raw.number_of_nodes()
print (n_lcc)

print ("diagonoal augmentation")
G = cvec.diag_aug(G_lcc)


sorted_vertex = sorted(G.nodes())
A = nx.to_scipy_sparse_matrix(G, nodelist=sorted_vertex)

cvec = G.degree()


print ("compute ase")
max_dim_ase = int(n_lcc/2)
#ASE = graph_ase_BNU.adjacency_embedding(G, max_dim=max_dim_ase, get_lcc=False, elb=1, svd_seed = 1234)

if 1: 
    max_dim = max_dim_ase
    elb = 1
    svd_seed = 1234
    U, Sigma, VT = randomized_svd(A, 
                              n_components=min(max_dim, n - 1),
                              n_iter=50,
                              random_state=svd_seed)

    ##print ("dimension reduction (elbow selection)")
    #rank_graph =  getElbows_BNU.getElbows(Sigma, n_elbows=elb)
    
    #reduced_dim = rank_graph[(elb-1)]
    ##print ("elbow is %d" %reduced_dim)

    reduced_dim = 2 #i keep getting 1 here! 
    s_sqrt = np.sqrt(Sigma) 

    
    s_sqrt_dim_reduced = s_sqrt[:reduced_dim]
    U_dim_reduced = U[:, :reduced_dim ]
    VT_dim_reduced =VT[:reduced_dim, :]

    Xhat_ase = np.multiply( s_sqrt_dim_reduced, U_dim_reduced)
          

max_k_ase = 2
ASE_cluster, ASE_BIC_score = gmm_BNU.gaussian_clustering(Xhat_ase, max_clusters = max_k_ase, min_clusters=2)




####
####
#LSE
####
####
if 1: 
    if nx.is_directed(G) == False:
        deg = (A.sum(axis=1).T).astype(float) 
        deg_array = np.squeeze(np.asarray(deg))
        D = np.diag(deg_array**(-0.5))
        LSE_Matrix = D @ A @ D 

    else:    
        deg = (A.sum(axis=1).T + A.sum(axis=0)).astype(float) 
        deg_array = np.squeeze(np.asarray(deg))
        D = np.diag(deg_array**(-1))
        LSE_Matrix = np.identity(n) - D @ A  


    #print ("spectral embedding into %d dimensions" %max_dim)
    U, Sigma, VT = randomized_svd(LSE_Matrix, 
                              n_components=min(max_dim, n - 1),
                              n_iter=50,
                              random_state=svd_seed)

    #print ("dimension reduction (elbow selection)")
    #rank_graph =  getElbows_BNU.getElbows(Sigma, n_elbows=elb)
    #reduced_dim = rank_graph[(elb-1)]

    reduced_dim = 2
    #print ("elbow is %d" %reduced_dim)
    s_sqrt = np.sqrt(Sigma) #[np.newaxis] Zeinab commented this out

    s_sqrt_dim_reduced = s_sqrt[:reduced_dim]
    U_dim_reduced = U[:, :reduced_dim ]
    VT_dim_reduced =VT[:reduced_dim, :]

    Xhat_lse = np.multiply( s_sqrt_dim_reduced, U_dim_reduced)
          

max_k_lse = 2
LSE_cluster, LSE_BIC_score = gmm_BNU.gaussian_clustering(Xhat_lse, max_clusters = max_k_lse, min_clusters=2)







#cnf_matrix = confusion_matrix(vertex_labels, ASE_cluster)
#print (cnf_matrix)





Ytrue = vertex_labels
print ('ASE Prediction')

Ypred = ASE_cluster
classes = list(set(Ytrue))
n_class = len(classes)

# error = np.array([list(zip(Ytrue,Ypred).count(x)) for x in itertools.product(classes,repeat=2)]).reshape(n_class,n_class)
# print (error)

error = np.array([z.count(x) for z in [list(zip(Ytrue,Ypred))] for x in itertools.product(classes,repeat=2)]).reshape(n_class,n_class)
print (error[:, :2])




Ytrue = vertex_labels
print ('LSE Prediction')

Ypred = LSE_cluster
classes = list(set(Ytrue))
n_class = len(classes)

# error = np.array([list(zip(Ytrue,Ypred).count(x)) for x in itertools.product(classes,repeat=2)]).reshape(n_class,n_class)
# print (error)

error = np.array([z.count(x) for z in [list(zip(Ytrue,Ypred))] for x in itertools.product(classes,repeat=2)]).reshape(n_class,n_class)
print (error[:, :2])




#########   

#pdb.set_trace()


# uniqueT, countsT = np.unique(Ytrue, return_counts=True) 
# Ytrue_counts = dict(zip(uniqueT, countsT))

Ytrue_ase_block1 = np.array(Ytrue)[ASE_cluster==0]
uniqueA, countsA  = np.unique(Ytrue_ase_block1, return_counts=True)
Ytrue_ase_counts = dict(zip(uniqueA, countsA))


# Yhat_lse_block1 = np.array(Ytrue)[LSE_cluster==0]
# uniqueL, countsL  = np.unique(Yhat_lse_block1, return_counts=True)
# Yhat_lse_counts = dict(zip(uniqueL, countsL))


#print ('True Labels')
#print ("{:<8} {:<15} ".format('Community','Counts'))
#for key, value in Ytrue_counts.items():
#	print ("{:<8} {:<15} ".format(key, value))


print ('ASE Labels')
print ("{:<8} {:<15} ".format('True Label','Pred: Block 0'))
for key, value in Ytrue_ase_counts.items():
	print ("{:<8} {:<15} ".format(key, value))


######### 

# df_cm = pd.DataFrame(cnf_matrix, range(K),
#                   range(K))
# #plt.figure(figsize = (10,7))
# sn.set(font_scale=1.4)#for label size
# sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
# output_fig2 = '/Users/z/Desktop/test2.png'
# plt.savefig(output_fig2, bbox_inches='tight')


