"""
Implement of EAGLE (agglomerativE hierarchicAl clusterinG based on maximaL cliquE) algorithm
for network community detection.

ref: Shen, H., et al. (2009). "Detect overlapping and hierarchical community structure in networks."
        Physica A: Statistical Mechanics and its Applications 388(8): 1706-1712.
ref: Shen, H.-W. (2013). Community structure of complex networks, Springer Science & Business Media.


Hao Zhu
16.12.2020
"""


import numpy as np
import networkx as nx
import copy
import os
import sys


class EAGLE:
    """
    Class for EAGLE (agglomerativE hierarchicAl clusterinG based on maximaL cliquE) algorithm.
    """
    def __init__(self, graph_data, data_type, maximal_clique_cutoff=3):
        """Initialize a EAGLE algorithm with a graph and parameters.

        Parameters
        ----------
        graph_data : NetworkX graph or numpy.ndarray
                    data containing graph structure.
        data_type : 'networkx_graph' or 'adjacency_matrix'
                    data type of graph_data.
        maximal_clique_cutoff : int, (default=3)
                                clique with nodes less than maximal_clique_cutoff will be ignored.
        """
        if data_type == 'networkx_graph':
            self._g = graph_data
            self._adj_matrix = nx.adjacency_matrix(graph_data).todense()
        elif data_type == 'adjacency_matrix':
            self._g = self._set_graph_from_adjmatrix(graph_data)
            self._adj_matrix = graph_data
        else:
            raise ValueError("Unknown graph data type '{}'. "
                             "Data type should be in ['networkx_graph', 'adjacency_matrix']".format(data_type))
        self._remove_zero_degree_nodes()

        self._maximal_clique_cutoff = maximal_clique_cutoff

        self._dendrogram = {}
        self._EQ_list = None
        self._max_EQ_index = None
        self._layer_with_max_EQ = None

    def _set_graph_from_adjmatrix(self, adj_matrix):
        """Generate a NetworkX graph according to its adjacency matrix.

        Parameters
        ----------
        adj_matrix : numpy.ndarray
                    adjacency matrix of graph.

        Returns
        -------
        graph: NetworkX graph
            A NetworkX graph data generated based on the input adjacency matrix.
        """
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        num_nodes = adj_matrix.shape[0]
        graph = nx.Graph()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i < j and adj_matrix[i, j] == 1:
                    graph.add_edge(i+1, j+1)

        return graph

    def _remove_zero_degree_nodes(self):
        """Remove those nodes without any edges in the graph.

        """
        for node, degree in self._g.degree:
            if degree == 0:
                self._g.remove_node(node)

    def _find_maximal_cliques(self):
        """Generate a list containing all the basic cliques for clustering.

        Step 1. Find all the maximal cliques within the graph.
        Step 2. Duplicate those cliques with nodes less than maximal_clique_cutoff.
        Step 3. All the remainder nodes are regarded as a single clique.

        Returns
        -------
        basic_cliques: list
                    A list containing all the cliques generated according to the steps above.
        """
        maximal_cliques = nx.find_cliques(self._g)
        basic_cliques = []
        remainder_nodes = list(self._g.nodes)
        for clique in maximal_cliques:
            if len(clique) >= self._maximal_clique_cutoff:
                basic_cliques.append(clique)
                for n in remainder_nodes:
                    if n in clique:
                        remainder_nodes.remove(n)
        if len(remainder_nodes) != 0:
            basic_cliques.append(remainder_nodes)

        return basic_cliques

    def _similarity(self, community_1, community_2):
        """Calculate the similarity between two communities.

        Similarity (S) is calculated according to:

            S = \frac{1}{2m} \sum_{v \in C_1, w \in C_2, v \neq w}[A_{vw} - \frac{k_v k_w}{2m}]

        where A_{vw} is the element of adjacency matrix of the network (only undirected unweighted graphs are
        considered), which takes value 1 if there is an edge between vertex v and vertex w and 0 otherwise.
        m = \frac{1}{2} \sum_{vw} A_{vw} is the total number of edges in the network.
        k_v is the degree of node v.

        Parameters
        ----------
        community_1 : list
                    A list containing nodes of community 1.
        community_2 : list
                    A list containing nodes of community 2.

        Returns
        -------
        S: float
            Similarity between community_1 and community_2.
        """
        m = 0.5 * np.sum(self._adj_matrix)
        S = 0
        for i in community_1:
            for j in community_2:
                if i != j:
                    ki = self._g.degree(i)
                    kj = self._g.degree(j)
                    Aij = self._adj_matrix[i-1, j-1]
                    S += Aij - ki*kj/2/m
        S = S/2/m

        return S

    def _clustering(self):
        """Perform the first stage of EAGLE algorithm -- generate a dendrogram.

        First select the pair of communities with the maximum similarity, incorporate them into a new one
        and calculate the similarity between the new community and other communities. Then repeat this process
        until only one community remains.

        """
        basic_cliques = self._find_maximal_cliques()
        num_cliques = len(basic_cliques)

        S_matrix = np.zeros((num_cliques, num_cliques))
        print('Building the dendrogram.')
        for i in reversed(range(1, num_cliques + 1)):
            if i == num_cliques:
                self._dendrogram[i] = copy.deepcopy(basic_cliques)

                # Calculate the similarity matrix between every two communities
                # in the basic cliques.
                for k, c1 in enumerate(basic_cliques):
                    for l, c2 in enumerate(basic_cliques):
                        if k > l:
                            S_matrix[k - 1, l - 1] = self._similarity(c1, c2)
                        elif k == l:
                            S_matrix[k - 1, l - 1] = float('-inf')
            else:
                new_layer = copy.deepcopy(self._dendrogram[i + 1])

                # Get the index of two cliques with max similarity.
                c1_idx, c2_idx = divmod(S_matrix.argmax(), S_matrix.shape[0])

                # Combine the two cliques with max similarity and append to the end
                # of the list containing the community structure of this layer.
                new_layer.append(list(set(new_layer[c1_idx] + new_layer[c2_idx])))

                # Pop the two cliques which are combined into the new clique
                # from the list containing the community structure
                # and from the similarity matrix.
                new_layer.pop(c1_idx)
                new_layer.pop(c2_idx)
                self._dendrogram[i] = new_layer

                S_matrix = np.delete(S_matrix, c1_idx, axis=0)
                S_matrix = np.delete(S_matrix, c1_idx, axis=1)
                S_matrix = np.delete(S_matrix, c2_idx, axis=0)
                S_matrix = np.delete(S_matrix, c2_idx, axis=1)

                # Calculate the similarity between the new clique and all others remained
                # cliques and stack to the similarity matrix.
                S_matrix = np.hstack((S_matrix, np.zeros((S_matrix.shape[0], 1))))
                S_matrix = np.vstack((S_matrix, np.zeros((1, S_matrix.shape[1]))))

                for c_idx, c in enumerate(new_layer[:-1]):
                    S_matrix[-1, c_idx] = self._similarity(c, new_layer[-1])

            print('\r Remain/Total Iterations: {}/{}'.format(i - 1, num_cliques), end='')
            sys.stdout.flush()

    def _get_node_belonging(self, communities):
        """Calculate the number of community that each node in the graph belong to.

        Parameters
        ----------
        communities : list
                    A list containing the community structure, where each community is
                    represented using a list containing the nodes of this community.

        Returns
        -------
        node_belonging_dic: dict, (keys = node_index, value = number of community the node belongs to)
                        A dict containing the number of community that each node in the
                        graph belong to.
        """
        node_belonging_dic = {}
        for node in self._g.nodes:
            n = 0
            for community in communities:
                if node in community:
                    n += 1
            node_belonging_dic[node] = n
        return node_belonging_dic

    def _extended_modularity(self, communities, node_belonging_dic):
        """Calculate the extended modularity of a community division pattern.

        Extended modularity (EQ) is calculated according to:


            EQ = \frac{1}{2m} \sum_i \sum_{v \in C_i, w \in C_i}\frac{1}{O_v O_w}[A_{vw} - \frac{k_v k_w}{2m}]

        where O_v is the number of communities to which node v belongs.

        Parameters
        ----------
        communities : list
                    A list containing the community structure, where each community is
                    represented using a list containing the nodes of this community.
        node_belonging_dic : dict, (keys = node_index, value = number of community the node belongs to)
                            A dict containing the number of community that each node in the
                            graph belong to.

        Returns
        -------
        EQ: float
            extended modularity of the given community division pattern.
        """
        m = 0.5 * np.sum(self._adj_matrix)

        EQ = 0

        for community in communities:
            for i in community:
                for j in community:
                    Oi = node_belonging_dic[i]
                    Oj = node_belonging_dic[j]
                    ki = self._g.degree(i)
                    kj = self._g.degree(j)
                    Aij = self._adj_matrix[i - 1, j - 1]
                    EQ += (Aij - ki * kj / 2 / m) / Oi / Oj
        EQ = EQ / 2 / m
        return EQ

    def _cut(self):
        """Perform the second stage of EAGLE algorithm -- cut the dendrogram.

        First calculate all the EQ under the community division pattern given by the dendrogram.
        (Each layer of the dendrogram represents a division pattern.)
        Then cut the dendrogram at the layer with the maximum EQ.

        Returns
        -------
        EQ_list: list
                A list containing all the EQ under each layer of the dendrogram.
        max_EQ_index: int
                    The index of EQ_list where EQ is maximal.
        """
        num_layers = len(self._dendrogram)
        EQ_list = []
        print('\nFinding maximal EQ.')
        for i in range(1, num_layers + 1):
            layer = self._dendrogram[i]
            node_belonging_dic = self._get_node_belonging(layer)
            EQ = self._extended_modularity(layer, node_belonging_dic)
            EQ_list.append(EQ)

            print('\r Remain/Total Iterations: {}/{}'.format(num_layers - i, num_layers), end='')
            sys.stdout.flush()

        max_EQ = max(EQ_list)
        max_EQ_index = EQ_list.index(max_EQ) + 1

        return EQ_list, max_EQ_index

    def fit(self):
        """Perform the EAGLE algorithm.

        This function is the only function need to be called when performing the EAGLE algorithm
        after the class EAGLE is instantiated.

        """
        self._clustering()
        self._EQ_list, self._max_EQ_index = self._cut()
        self._layer_with_max_EQ = self._dendrogram[self._max_EQ_index]
        print('\nDone.')

    def get_community_structure(self):
        """Get the detected community structure via EAGLE algorithm.

        Returns
        -------
        layer_with_max_EQ: list
                        A list containing the detected community structure, where each community is
                        represented using a list containing the nodes of this community.
        """
        return self._layer_with_max_EQ

    def get_EQ_list(self):
        """Get the list containing all the EQ under each layer of the dendrogram.

        Returns
        -------
        EQ_list: list
                A list containing all the EQ under each layer of the dendrogram.
        """
        return self._EQ_list

    def get_dendrogram(self):
        """Get the dendrogram generated in the first step of EAGLE algorithm.

        Returns
        -------
        dendrogram: dict (key = layer number, value = community structure of the layer)
                    A dendrogram containing all the community structure divided with
                    different pattern along the clustering process.
        """
        return self._dendrogram

    def save_community_structure(self, path='./'):
        """Save the detected community structure via EAGLE algorithm.

        * When loading the .npy file using numpy.load(), parameter 'allow_pickle' need to be set to True.
        Parameters
        ----------
        path : string
            Saving path.

        """
        np.save(os.path.join(path, 'community_structure.npy'), np.array(self._layer_with_max_EQ, dtype=object))

    def save_dendrogram(self, path='./'):
        """Save the dendrogram generated in the first step of EAGLE algorithm.

        * When loading the .npy file using numpy.load(), parameter 'allow_pickle' need to be set to True.
        Parameters
        ----------
        path : string
            Saving path.

        """
        np.save(os.path.join(path, 'dendrogram.npy'), np.array(self._dendrogram, dtype=object))

    def save_EQ_list(self, path='./'):
        """Save the list containing all the EQ under each layer of the dendrogram.

        * When loading the .npy file using numpy.load(), parameter 'allow_pickle' need to be set to True.
        Parameters
        ----------
        path : string
            Saving path.

        """
        np.save(os.path.join(path, 'EQ_list.npy'), np.array(self._EQ_list))

    def graph_info(self):
        """
        Print info of the graph, including number of nodes and edges.
        """
        print(nx.info(self._g))

    def community_info(self):
        """
        Print info of the community structure, including number of communities and its EQ.
        """
        print("{} communities detected with maximal EQ of {:.3f}.".format(
            self._max_EQ_index, self._EQ_list[self._max_EQ_index - 1]
        ))


