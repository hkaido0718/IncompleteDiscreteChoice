import itertools
import numpy as np
import cvxpy as cp
import scipy
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import differential_evolution, LinearConstraint, NonlinearConstraint

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
    data (list): List containing two arrays [Y, X]
    seed (int, optional): Seed for the random number generator

    Returns:
    tuple: Two lists, each containing two arrays representing the two halves of the data
    """
    Y, X = data
    n = Y.shape[0]

    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(n)
    np.random.shuffle(indices)

    split_point = n // 2
    indices_1 = indices[:split_point]
    indices_2 = indices[split_point:]

    Y0 = Y[indices_1]
    X0 = X[indices_1]
    Y1 = Y[indices_2]
    X1 = X[indices_2]

    data0 = [Y0, X0]
    data1 = [Y1, X1]

    return data0, data1

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

    def plot_graph(self, pos=None, title=''):
        # Determine node positions if not provided
        if pos is None:
            pos = nx.drawing.layout.bipartite_layout(self.B, self.U_nodes,align='horizontal')
        
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
        return calculate_Ftheta(X_supp[i], theta)

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

def calculate_p0(theta, data, gmodel, calculate_Ftheta):
    """
    Calculate the p0 and nutheta for each unique X value.

    Parameters:
    theta (np.array): Parameter vector.
    data (tuple): Tuple containing Y and X arrays.
    gmodel (BipartiteGraph): Instance of the BipartiteGraph class.
    calculate_Ftheta (function): Function to calculate Ftheta.

    Returns:
    tuple: (p_events, nutheta, p0)
    p_events (list): List of subset probabilities for each unique X value.
    nutheta (list): List of sharp lower bounds for each unique X value.
    p0 (list): List of solutions to the linear feasibility problem for each unique X value.
    """
    Y, X = data
    Y_nodes = gmodel.Y_nodes
    U_nodes = gmodel.U_nodes
    B = gmodel.B
    tolcon = 1e-4

    # Step 1: Obtain X_supp
    _, _, _, X_supp = calculate_ccp(Y,X,Y_nodes)
    Nx = len(X_supp)


    # Step 3: Compute Ftheta at \(\theta\)
    Nu = len(U_nodes)
    Ftheta = np.zeros((Nx, Nu))
    for i in range(Nx):
        Ftheta[i, :] = calculate_Ftheta(X_supp[i], theta)

    # Step 4: Compute \(\nu_{\theta}\) and find p for each i
    nutheta = []
    p0 = []
    for i in range(Nx):
        results, sharp_lower_bounds = gmodel.calculate_sharp_lower_bound(Ftheta[i])

        # Append sharp_lower_bounds to nutheta
        nutheta.append(sharp_lower_bounds)

        # Linear feasibility problem
        filtered_results = [result for result in results if result[1]]
        num_rows = len(filtered_results)
        num_cols = len(Y_nodes)
        A = np.zeros((num_rows, num_cols), dtype=int)
        b = np.zeros(num_rows)

        for j, (subset_set, exclusive_u_nodes, total_prob) in enumerate(filtered_results):
            for k, y_node in enumerate(Y_nodes):
                if y_node in subset_set:
                    A[j, k] = 1
            b[j] = total_prob - tolcon

        # Solve the linear feasibility problem Ax >= b with x in the probability simplex
        c = np.zeros(num_cols)  # Dummy objective function
        bounds = [(tolcon, 1-tolcon) for _ in range(num_cols)]  # Encourage interior solutions by setting bounds to be slightly within (0,1)

        A_eq = np.ones((1, num_cols))
        b_eq = np.array([1])

        res = scipy.optimize.linprog(c, A_ub=-A, b_ub=-b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            p0.append(res.x)
        else:
            print(f"No feasible solution exists for X index {i}.")

    return nutheta, p0

def calculate_qtheta(theta, data, gmodel, calculate_Ftheta, p0):
    """
    Calculate the qtheta for each unique X value using CVXPY.

    Parameters:
    theta (np.array): Parameter vector.
    data (tuple): Tuple containing Y and X arrays.
    gmodel (BipartiteGraph): Instance of the BipartiteGraph class.
    calculate_Ftheta (function): Function to calculate Ftheta.
    p0 (list): List of solutions to the linear feasibility problem for each unique X value.

    Returns:
    list: qtheta for each unique X value.
    """
    Y, X = data
    Y_nodes = gmodel.Y_nodes
    U_nodes = gmodel.U_nodes
    B = gmodel.B

    Ny = len(Y_nodes)
    tolcon = 1e-3

    # Step 1: Obtain X_supp
    _, _, _, X_supp = calculate_ccp(Y,X,Y_nodes)
    Nx = len(X_supp)

    # Step 3: Compute Ftheta at \(\theta\)
    Nu = len(U_nodes)
    Ftheta = np.zeros((Nx, Nu))
    for i in range(Nx):
        Ftheta[i, :] = calculate_Ftheta(X_supp[i], theta)

    qtheta = []

    for i in range(Nx):
        p = p0[i]

        # Setup constraints
        results, _ = gmodel.calculate_sharp_lower_bound(Ftheta[i])
        filtered_results = [result for result in results if result[1]]
        num_rows = len(filtered_results)
        A = np.zeros((num_rows, Ny), dtype=int)
        b = np.zeros(num_rows)

        for j, (subset_set, exclusive_u_nodes, total_prob) in enumerate(filtered_results):
            for k, y_node in enumerate(Y_nodes):
                if y_node in subset_set:
                    A[j, k] = 1
            b[j] = total_prob - tolcon

        # Define the optimization variables and constraints
        q = cp.Variable(Ny)
        constraints = [q >= tolcon, q <= 1-tolcon, cp.sum(q) == 1]  # Probability simplex and bounds constraints
        for j in range(num_rows):
            constraints.append(A[j, :] @ q >= b[j])

        # Define the objective function
        objective = cp.Minimize(cp.sum(cp.rel_entr(q + p, q)))

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status == cp.OPTIMAL:
            qtheta.append(q.value)
        else:
            print(f"No feasible solution exists for X index {i}.")
            qtheta.append(np.zeros(Ny))

    return qtheta

def calculate_L1(data,gmodel, p0, truncation_threshold=1e10):
    """
    Calculate the lnL1 value for p0.

    Parameters:
    data (tuple): Tuple containing Y and X arrays.
    gmodel (BipartiteGraph): Instance of the BipartiteGraph class.
    p0 (list): List of solutions to the linear feasibility problem for each unique X value.
    truncation_threshold (float): The value at which to truncate lnLR to ensure it stays finite.

    Returns:
    float: The calculated lnL1 value.
    """
    Y, X = data
    # Compute ccp_array and Px
    _, ccp_array, Px, _ = calculate_ccp(Y, X, gmodel.Y_nodes)

    Nx,Ny = ccp_array.shape

    # Compute weights w and count
    n = len(Y)
    w = np.repeat(Px, Ny).reshape(Nx, Ny)
    count = n * ccp_array * w

    sumlnL1 = np.sum(np.log(p0)*count)
    return sumlnL1


def calculate_L0(theta, data, gmodel, calculate_Ftheta, p0, truncation_threshold=1e10):
    """
    Calculate the lnL0 value for the given theta.

    Parameters:
    theta (np.array): Parameter vector.
    data (tuple): Tuple containing Y and X arrays.
    gmodel (BipartiteGraph): Instance of the BipartiteGraph class.
    calculate_Ftheta (function): Function to calculate Ftheta.
    p0 (list): List of solutions to the linear feasibility problem for each unique X value.
    truncation_threshold (float): The value at which to truncate lnLR to ensure it stays finite.

    Returns:
    float: The calculated lnL0 value.
    """
    Y, X = data
    Nx = len(p0)
    Ny = len(gmodel.Y_nodes)

    # Calculate qtheta
    qtheta = calculate_qtheta(theta, data, gmodel, calculate_Ftheta, p0)

    # Compute log-likelihood 
    lnL0 = np.zeros((Nx, Ny))
    for i in range(Nx):
          lnL0[i, :] = np.log(qtheta[i])
          lnL0[i, :] = np.clip(lnL0[i, :], -truncation_threshold, truncation_threshold)  # Truncate to threshold

    # Compute ccp_array, Px, and X_supp
    _, ccp_array, Px, X_supp = calculate_ccp(Y, X, gmodel.Y_nodes)

    # Compute weights w and count
    n = len(Y)
    w = np.repeat(Px, Ny).reshape(Nx, Ny)
    count = n * ccp_array * w

    # Calculate T
    sumlnL0 = np.sum(lnL0 * count)

    return sumlnL0

def calculate_LR(data, gmodel, calculate_Ftheta, LB, UB, linear_constraint=None, nonlinear_constraint=None, seed=123, split=None):
    """
    Calculate the T value for the given parameters.

    Parameters:
    data (list): List containing Y and X arrays
    gmodel (BipartiteGraph): Instance of the BipartiteGraph class
    calculate_Ftheta (function): Function to calculate Ftheta
    LB (list): Lower bounds for theta
    UB (list): Upper bounds for theta
    linear_constraint (LinearConstraint, optional): Linear constraint
    nonlinear_constraint (NonlinearConstraint, optional): Nonlinear constraint
    seed (int, optional): Seed for the random number generator (default is 123)
    split (str, optional): If "swap", swap the roles of data0 and data1; if "crossfit", calculate T and T^swap and return T^crossfit

    Returns:
    float: The calculated T, T^swap, or T^crossfit value
    """
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Calculate bounds from LB and UB
    bounds = [(low, high) for low, high in zip(LB, UB)]

    def calculate_single_T(data0, data1, label):
        # Step 1: Define the function to minimize for Qhat
        def objective_function_Qhat(theta):
            return calculate_Qhat(theta, data1, gmodel, calculate_Ftheta)

        # Perform the optimization for Qhat
        result = differential_evolution(objective_function_Qhat, bounds, seed=seed)
        thetahat1 = result.x
        min_Qhat = result.fun

        print(f"Initial Estimator{label}:", thetahat1)
        print("Minimum Qhat:", min_Qhat)

        # Step 2: Calculate p0 and unrestricted log-likelihood
        _, p0 = calculate_p0(thetahat1, data0, gmodel, calculate_Ftheta)
        sumlnL1 = calculate_L1(data0, gmodel, p0)
        print("Unrestricted log-likelihood:", sumlnL1)

        # Define the function to minimize for L0
        def objective_function_L0(theta):
            return -calculate_L0(theta, data0, gmodel, calculate_Ftheta, p0)

        # Callback function to print intermediate results
        def callback(xk, convergence):
            print(f"Current Qhat: {objective_function_L0(xk)}")
            print(f"Convergence: {convergence}")

        # Set the constraints for the optimization
        constraints = []
        if linear_constraint is not None:
            constraints.append(linear_constraint)
        if nonlinear_constraint is not None:
            constraints.append(nonlinear_constraint)

        # Perform the optimization for L0
        result = differential_evolution(
            objective_function_L0, 
            bounds, 
            constraints=constraints, 
            #callback=callback, 
            seed=seed
        )
        thetahat0 = result.x
        sumlnL0 = -result.fun

        print(f"RMLE{label}:", thetahat0)
        print("Restricted log-likelihood:", sumlnL0)
        T = np.exp(sumlnL1 - sumlnL0)
        print(f"T{label}:", T)
        return T

    # Split the data
    data0, data1 = split_data(data, seed=seed)

    if split == "swap":
        data0, data1 = data1, data0
        T_swap = calculate_single_T(data0, data1, "^swap")
        return T_swap

    elif split == "crossfit":
        T = calculate_single_T(data0, data1, "")
        T_swap = calculate_single_T(data1, data0, "^swap")
        T_crossfit = (T + T_swap) / 2
        print("T^crossfit:", T_crossfit)
        return T_crossfit

    else:
        T = calculate_single_T(data0, data1, "")
        return T