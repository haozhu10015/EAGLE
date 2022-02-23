import numpy as np
from eagle import EAGLE
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    io_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example')

    # Load the adjacency matrix of the graph.
    adj_matrix = np.loadtxt(os.path.join(io_path, 'adjacency_matrix.csv'), delimiter=',')

    # Instantiate EAGLE algorithm using the adjacency matrix.
    eagle = EAGLE(adj_matrix, data_type='adjacency_matrix', maximal_clique_cutoff=3)
    eagle.graph_info() # Print the graph info.

    # Perform EAGLE algorithm.
    eagle.fit()

    # Save the outputs.
    eagle.save_dendrogram(io_path)
    eagle.save_EQ_list(io_path)
    eagle.save_community_structure(io_path)

    # Print info of the detected community structure.
    eagle.community_info()

    for i, community in enumerate(eagle.get_community_structure()):
        print("Community-{} ({} nodes): ".format(i + 1, len(community)), end='')
        print(community)

    # Plot the EQ list.
    max_EQ_index = len(eagle.get_community_structure())

    fig = plt.figure(10, dpi=300)
    plt.plot(range(1, len(eagle.get_EQ_list()) + 1), eagle.get_EQ_list())
    plt.plot([max_EQ_index, max_EQ_index], [0, 0.55], ls='--')
    plt.ylabel('EQ')
    fig.savefig(os.path.join(io_path, 'EQ_list.png'))
    plt.show()

