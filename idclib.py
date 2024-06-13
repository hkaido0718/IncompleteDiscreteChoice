import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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


def calculate_ccp(X_vals, Y):
    """
    Calculate the conditional choice probabilities (CCP) for Y given X.

    Parameters:
    X_vals (np.ndarray): Array of X values of shape (n, d_X)
    Y (np.ndarray): Array of Y values of shape (n, d_Y)

    Returns:
    dict: Conditional frequencies of Y given X
    np.ndarray: Array of conditional choice probabilities for each unique X value
    """
    # Find unique X values
    unique_x_vals = np.unique(X_vals, axis=0)
    
    # Calculate the shape of the Y array
    d_Y = Y.shape[1]
    
    # Initialize an empty list to store the probabilities
    probabilities = []

    # Initialize a list to store the unique X values in tuple format for later reporting
    unique_x_list = []

    # Iterate over each unique X value
    for x in unique_x_vals:
        # Add the unique X value to the list for reporting
        unique_x_list.append(tuple(x))
        
        # Create a mask to filter rows where X equals the current unique X value
        mask = (X_vals == x).all(axis=1)
        
        # Filter Y values corresponding to the current unique X value
        y_given_x = Y[mask]
        
        # Initialize a dictionary to count occurrences of each Y value as tuples
        counts = {tuple(y): 0 for y in itertools.product([0, 1], repeat=d_Y)}
        
        # Count occurrences of each Y value
        for y in y_given_x:
            counts[tuple(y)] += 1
        
        # Calculate the total number of occurrences
        total = sum(counts.values())
        
        # Calculate the conditional probabilities
        ccp = [counts[tuple(y)] / total for y in itertools.product([0, 1], repeat=d_Y)]
        
        # Append the CCP to the probabilities list
        probabilities.append(ccp)

    # Convert the list of probabilities to a NumPy array
    ccp_array = np.array(probabilities).reshape(-1, len(counts))

    return unique_x_vals, ccp_array


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
    Y_nodes = gmodel.Y_nodes
    U_nodes = gmodel.U_nodes

    # Step 1: Compute ccp
    unique_x_vals, ccp_array = calculate_ccp(X, Y)
    Nx, Ny = ccp_array.shape

    # Step 2: Compute \(\hat{p}(A|x)\)
    p_events = []
    for i in range(Nx):
        results, subset_probabilities = calculate_subset_probabilities(ccp_array[i], Y_nodes)
        p_events = np.append(p_events, subset_probabilities)

    # Step 3: Compute Ftheta at \(\theta\)
    Nu = len(U_nodes)
    Ftheta = np.zeros((Nx, Nu))
    for i in range(Nx):
        Ftheta[i, :] = calculate_Ftheta(unique_x_vals[i, :], theta)

    # Step 4: Compute \(\nu_{\theta}\)
    nutheta = []
    for i in range(Nx):
        results, sharp_lower_bounds = gmodel.calculate_sharp_lower_bound(Ftheta[i])
        nutheta = np.append(nutheta, sharp_lower_bounds)

    # Step 5: Compute \(\hat{Q}(\theta)\)
    difference = nutheta - p_events
    diff_pos = np.maximum(difference, 0)
    hatQ = np.max(diff_pos)
    return hatQ


