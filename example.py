import numpy as np
from eagle import EAGLE


if __name__ == '__main__':
    # Load the adjacency matrix of the graph.
    adj_matrix = np.loadtxt('./example/adjacency_matrix.csv', delimiter=',')

    # Instantiate EAGLE algorithm using the adjacency matrix.
    eagle = EAGLE(adj_matrix, data_type='adjacency_matrix', maximal_clique_cutoff=3)
    eagle.graph_info() # Print the graph info.

    # Perform EAGLE algorithm.
    eagle.fit()

    # Save the outputs.
    eagle.save_dendrogram('./example/')
    eagle.save_EQ_list('./example/')
    eagle.save_community_structure('./example/')

    # Print info of the detected community structure.
    eagle.community_info()

