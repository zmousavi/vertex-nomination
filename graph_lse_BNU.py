import numpy as np
import networkx as nx
import scipy
from sklearn.utils.extmath import randomized_svd


import getElbows_BNU 
#import lcc_BNU
import collections

#testing
    
def laplacian_embedding(G, max_dim=2, elb=1, get_lcc=True, weightcol='weight', svd_seed=None):

    """
    Inputs
        G - A networkx graph
    Outputs
        eig_vectors - The scaled (or unscaled) eigenvectors
    """
    # if get_lcc==True:
    #     #print("extracting largest_connected_component")
    #     G_lcc = lcc_BNU.extract_lcc(G)
    # else:
    #     G_lcc = G.copy()

    # weightcolumn = weightcol
    
    # print("pass_to_ranks")
    # G_ptr = ptr.pass_to_ranks(G_lcc, weightcol=weightcolumn)
    
    # print ("diagonoal augmentation")
    # G_aug_ptr= cvec.diag_aug(G_ptr, weightcol=weightcolumn)

    sorted_vertex = sorted(G.nodes())
    A = nx.to_scipy_sparse_matrix(G, nodelist=sorted_vertex)

    row, col = A.shape
    n = min(row, col)

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
    rank_graph =  getElbows_BNU.getElbows(Sigma, n_elbows=elb)
    reduced_dim = rank_graph[(elb-1)]

    #print ("elbow is %d" %reduced_dim)
    s_sqrt = np.sqrt(Sigma) #[np.newaxis] Zeinab commented this out

    s_sqrt_dim_reduced = s_sqrt[:reduced_dim]
    U_dim_reduced = U[:, :reduced_dim ]
    VT_dim_reduced =VT[:reduced_dim, :]

    Xhat1 = np.multiply( s_sqrt_dim_reduced, U_dim_reduced)
          
    if nx.is_directed(G) == False:
        Xhat2 = np.array([]).reshape(Xhat1.shape[0],0)
    else:
        Xhat2 = np.multiply( np.transpose(VT_dim_reduced), s_sqrt_dim_reduced)
    Xhat = np.concatenate((Xhat1, Xhat2), axis=1)
    
    embedded = collections.namedtuple('embedded',  'X vertex_labels')
    result = embedded(X = Xhat, vertex_labels = sorted_vertex)

    return result
