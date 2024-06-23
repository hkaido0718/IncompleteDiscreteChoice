import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def calculate_subset_probabilities(P0, Y_nodes):
    """
    Calculate the probabilities of all subsets of Y-nodes.

    Parameters:
    P0 (np.array): Array containing the probabilities for each Y-node.
    Y_nodes (list): List of Y-nodes.

    Returns:
    tuple: A tuple containing:
        - results (list): List of tuples with each tuple containing a subset of Y-nodes and their calculated probability.
        - subset_probabilities (np.array): Array of probabilities for each subset.
    """
    # Generate all subsets of Y_nodes
    all_subsets = []
    for r in range(len(Y_nodes) + 1):
        for subset in itertools.combinations(Y_nodes, r):
            all_subsets.append(subset)

    subset_probabilities = []  # List to store probabilities of each subset
    results = []  # List to store subsets and their probabilities

    # Calculate probability for each subset
    for subset in all_subsets:
        if len(subset) == 0:
            subset_prob = 0  # Probability of the empty set is 0
        elif len(subset) == len(Y_nodes):
            subset_prob = 1  # Probability of the entire set is 1
        else:
            subset_indices = [Y_nodes.index(node) for node in subset]  # Get indices of nodes in the subset
            subset_prob = np.sum([P0[i] for i in subset_indices])  # Sum probabilities of the nodes in the subset

        results.append((subset, subset_prob))  # Store the subset and its probability
        subset_probabilities.append(subset_prob)  # Append the subset probability to the list

    return results, np.array(subset_probabilities)  # Return results and subset probabilities as a NumPy array

def calculate_ccp(Y, X_vals, Y_nodes):
    unique_X_vals = np.unique(X_vals, axis=0)
    is_continuous = len(unique_X_vals) == len(X_vals)
    
    X_supp = sorted({tuple(x) for x in unique_X_vals})

    count_dict = {x: {y: 0 for y in Y_nodes} for x in X_supp}
    total_counts = {x: 0 for x in X_supp}

    for i in range(len(Y)):
        x = tuple(X_vals[i])
        if x in count_dict:
            y = tuple(Y[i])
            count_dict[x][y] += 1
            total_counts[x] += 1

    prob_dict = {}
    for x in count_dict:
        if total_counts[x] > 0:
            prob_dict[x] = {y: count_dict[x][y] / total_counts[x] for y in count_dict[x]}
        else:
            prob_dict[x] = {y: 0 for y in count_dict[x]}

    ordered_prob_dict = {tuple(X_vals[i]): prob_dict[tuple(X_vals[i])] for i in range(len(X_vals)) if tuple(X_vals[i]) in prob_dict}

    # Calculate the relative frequencies
    total_samples = len(Y)
    relative_frequencies = np.array([total_counts[x] / total_samples for x in X_supp])

    if is_continuous:
        # Create an np.array of conditional probabilities
        prob_array = np.array([[ordered_prob_dict[tuple(X_vals[i])][y] for y in Y_nodes] for i in range(len(X_vals))])
        return ordered_prob_dict, prob_array, relative_frequencies, X_vals
    else:
        rearranged_prob_array = np.array([[prob_dict[x][y] for y in Y_nodes] for x in X_supp])
        return ordered_prob_dict, rearranged_prob_array, relative_frequencies, X_supp


def split_data(data, seed=None):
    """
    Randomly split the data into two halves.

    Parameters:
    data (np.ndarray): Array containing the data [X, Y] of shape (n, 2)
    seed (int, optional): Seed for the random number generator

    Returns:
    tuple: Two numpy arrays representing the two halves of the data
    """
    if seed is not None:
        np.random.seed(seed)

    n = data.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    split_point = n // 2
    indices_1 = indices[:split_point]
    indices_2 = indices[split_point:]

    data_1 = data[indices_1]
    data_2 = data[indices_2]

    return data_1, data_2

class BipartiteGraph:
    def __init__(self, Y_nodes, U_nodes, edges):
        self.Y_nodes = Y_nodes
        self.U_nodes = U_nodes
        self.B = nx.DiGraph()
        self.B.add_nodes_from(Y_nodes, bipartite=0)  # Add Y-nodes with bipartite attribute
        self.B.add_nodes_from(U_nodes, bipartite=1)  # Add U-nodes with bipartite attribute
        self.B.add_edges_from(edges)  # Add directed edges

    def get_exclusive_u_nodes(self, subset_set):
        exclusive_u_nodes = set()
        for u_node in [node for node, attr in self.B.nodes(data=True) if attr['bipartite'] == 1]:
            neighbors = set(self.B.neighbors(u_node))
            if neighbors.issubset(subset_set):
                exclusive_u_nodes.add(u_node)
        return exclusive_u_nodes

    def sum_probabilities(self, exclusive_u_nodes, Ftheta):
        total_probability = 0.0
        for u_node in exclusive_u_nodes:
            index = self.U_nodes.index(u_node)
            total_probability += Ftheta[index]
        return total_probability

    def calculate_sharp_lower_bound(self, Ftheta):
        results = []
        for r in range(len(self.Y_nodes) + 1):
            for subset in itertools.combinations(self.Y_nodes, r):
                subset_set = set(subset)
                exclusive_u_nodes = self.get_exclusive_u_nodes(subset_set)
                total_prob = self.sum_probabilities(exclusive_u_nodes, Ftheta)
                results.append((subset_set, exclusive_u_nodes, total_prob))
        sharp_lower_bounds = np.array([result[2] for result in results]) # np.array([result[2] for result in results if result[1]])  # Filter out empty exclusive_u_nodes
        return results, sharp_lower_bounds

    def plot_graph(self, pos, title=''):
        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(self.B, pos, with_labels=True, labels={node: str(node) for node in self.B.nodes()},
                node_color=['lightblue'] * len(self.Y_nodes) + ['lightgreen'] * len(self.U_nodes),
                node_size=2000, font_size=12, font_color='black', edge_color='gray', arrows=True)
        
        # Add title and ensure equal aspect ratio
        plt.title(title)
        plt.axis('equal')
        plt.show()
    
def print_table(results):
    # Filter results to drop rows with empty exclusive_u_nodes
    filtered_results = [result for result in results if result[1]]
    
    # Calculate the maximum width for each column
    subset_width = max(len(str(result[0])) for result in filtered_results)
    exclusive_width = max(len(str(result[1])) for result in filtered_results)
    lower_bound_width = max(len(f"{result[2]:.2f}") for result in filtered_results)
    
    # Define minimum widths for each column
    min_subset_width = 20
    min_exclusive_width = 15
    min_lower_bound_width = 15

    # Adjust widths to be at least the minimum width
    subset_width = max(subset_width, min_subset_width)
    exclusive_width = max(exclusive_width, min_exclusive_width)
    lower_bound_width = max(lower_bound_width, min_lower_bound_width)
    
    # Add space between columns
    column_spacing = 4
    
    # Print the header
    header = f"{'Subset of Y-nodes':<{subset_width + column_spacing}} {'Exclusive U-nodes':<{exclusive_width + column_spacing}} {'Sharp Lower Bound':<{lower_bound_width + column_spacing}}"
    print(header)
    print("=" * len(header))
    
    # Print the filtered results
    for subset_set, exclusive_u_nodes, total_prob in filtered_results:
        print(f"{str(subset_set):<{subset_width + column_spacing}} {str(exclusive_u_nodes):<{exclusive_width + column_spacing}} {total_prob:<{lower_bound_width + column_spacing}.2f}")

def calculate_Qhat(theta, data, gmodel, calculate_Ftheta):
    Y, X = data
    n = Y.shape[0]
    Y_nodes = gmodel.Y_nodes
    U_nodes = gmodel.U_nodes

    # Step 1: Compute ccp
    _, ccp_array, Px, X_supp = calculate_ccp(Y, X, Y_nodes)
    Nx = len(X_supp)

    # Step 2: Compute \(\hat{p}(A|x)\)
    def compute_p_events(i):
        _, temp_p_events = calculate_subset_probabilities(ccp_array[i,:], Y_nodes)
        return temp_p_events

    with ThreadPoolExecutor() as executor:
        p_events = np.array(list(executor.map(compute_p_events, range(Nx))))

    # Step 3: Compute Ftheta at \(\theta\)
    def compute_Ftheta(i):
        return calculate_Ftheta(X_supp[i, :], theta)

    with ThreadPoolExecutor() as executor:
        Ftheta = np.array(list(executor.map(compute_Ftheta, range(Nx))))

    # Step 4: Compute \(\nu_{\theta}\)
    def compute_nutheta(i):
        _, temp_nutheta = gmodel.calculate_sharp_lower_bound(Ftheta[i])
        return temp_nutheta

    with ThreadPoolExecutor() as executor:
        nutheta = np.array(list(executor.map(compute_nutheta, range(Nx))))

    # Step 5: Compute \(\hat{Q}(\theta)\)
    difference = nutheta - p_events
    if np.unique(X, axis=0).shape[0] == n:
        meandiff = np.sum(difference, axis=0) / n
        Qhat = np.sum(np.maximum(meandiff, 0))
    else:
        diff_pos = np.maximum(difference, 0)
        J = difference.shape[1]
        w = np.repeat(Px,J).reshape(Nx,J)
        Qhat = np.sum(w*diff_pos) / n

    return Qhat


