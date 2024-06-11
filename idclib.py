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
        """
        Initialize the BipartiteGraph with Y-nodes, U-nodes, and edges.

        Parameters:
        Y_nodes (list): List of Y-nodes
        U_nodes (list): List of U-nodes
        edges (list of tuples): List of directed edges (U_node, Y_node)
        """
        self.Y_nodes = Y_nodes
        self.U_nodes = U_nodes
        self.B = nx.DiGraph()
        self.B.add_nodes_from(Y_nodes, bipartite=0)  # Add Y-nodes with bipartite attribute
        self.B.add_nodes_from(U_nodes, bipartite=1)  # Add U-nodes with bipartite attribute
        self.B.add_edges_from(edges)  # Add directed edges



def calculate_Qhat(theta, data, gmodel, calculate_Ftheta):
    Y, X = data
    Y_nodes = gmodel.Y_nodes
    U_nodes = gmodel.U_nodes
    B = gmodel.B

    # Step 1: Compute ccp
    unique_x_vals, ccp_array = calculate_ccp(X, Y)
    Nx, Ny = ccp_array.shape

    # Step 2: Compute \(\hat{p}(A|x)\)
    p_events = []
    for i in range(Nx):
        results, subset_probabilities = alculate_subset_probabilities(ccp_array[i], Y_nodes)
        p_events = np.append(p_events, subset_probabilities)

    # Step 3: Compute Ftheta at \(\theta\)
    Nu = len(U_nodes)
    Ftheta = np.zeros((Nx, Nu))
    for i in range(Nx):
        Ftheta[i, :] = calculate_Ftheta(unique_x_vals[i, :], theta)

    # Step 4: Compute \(\nu_{\theta}\)
    nutheta = []
    for i in range(Nx):
        results, sharp_lower_bounds = calculate_sharp_lower_bound(Y_nodes, U_nodes, B, Ftheta[i])
        nutheta = np.append(nutheta, sharp_lower_bounds)

    # Step 5: Compute \(\hat{Q}(\theta)\)
    difference = nutheta - p_events
    diff_pos = np.maximum(difference, 0)
    hatQ = np.max(diff_pos)
    return hatQ

