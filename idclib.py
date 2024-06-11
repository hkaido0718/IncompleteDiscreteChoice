import itertools
import numpy as np
import networkx as nx

def get_exclusive_u_nodes(G, A):
    """
    Returns a set of U-nodes that are connected only to the nodes in set A.

    Parameters:
    G (nx.DiGraph): The bipartite graph.
    A (set): A set of Y-nodes.

    Returns:
    set: A set of U-nodes that are connected only to the nodes in set A.
    """
    # Initialize an empty set for the U-nodes
    exclusive_u_nodes = set()

    # Iterate through all U-nodes in the graph
    for u_node in [node for node, attr in G.nodes(data=True) if attr['bipartite'] == 1]:
        # Get all neighbors of the U-node
        neighbors = set(G.neighbors(u_node))

        # Check if all neighbors are in the set A
        if neighbors.issubset(A):
            exclusive_u_nodes.add(u_node)

    return exclusive_u_nodes

def sum_probabilities(exclusive_u_nodes, U_nodes, Ftheta):
    """
    Returns the sum of the probabilities over the exclusive U-nodes.

    Parameters:
    exclusive_u_nodes (set): A set of exclusive U-nodes.
    U_nodes (list): List of all U-nodes in the order corresponding to Ftheta.
    Ftheta (np.array): Array containing the probabilities for each U-node.

    Returns:
    float: The sum of the probabilities over the exclusive U-nodes.
    """
    total_probability = 0.0

    for u_node in exclusive_u_nodes:
        # Find the index of the U-node in U_nodes
        index = U_nodes.index(u_node)
        # Add the corresponding probability to the total
        total_probability += Ftheta[index]

    return total_probability

def calculate_sharp_lower_bound(Y_nodes, U_nodes, B, Ftheta):
    """
    Calculate the sharp lower bounds for all subsets of Y-nodes.

    Parameters:
    Y_nodes (list): List of Y-nodes.
    U_nodes (list): List of U-nodes.
    B (nx.DiGraph): The bipartite graph.
    Ftheta (np.array): Array containing the probabilities for each U-node.

    Returns:
    tuple: A tuple containing:
        - results (list): List of tuples with subset of Y-nodes, exclusive U-nodes, and their sharp lower bound.
        - sharp_lower_bounds (np.array): Array of sharp lower bounds for each subset.
    """
    results = []

    # Generate all subsets of Y-nodes
    for r in range(len(Y_nodes) + 1):
        for subset in itertools.combinations(Y_nodes, r):
            subset_set = set(subset)  # Convert subset to a set for easier operations
            exclusive_u_nodes = get_exclusive_u_nodes(B, subset_set)  # Get U-nodes connected only to this subset
            total_prob = sum_probabilities(exclusive_u_nodes, U_nodes, Ftheta)  # Sum probabilities of these U-nodes
            results.append((subset_set, exclusive_u_nodes, total_prob))  # Store the result

    # Extract sharp lower bounds from results
    sharp_lower_bounds = np.array([result[2] for result in results])

    return results, sharp_lower_bounds


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