import numpy as np

def simulate_y_entrygame(n, beta1, beta2, delta1, delta2, rho, seed=None):
    """
    Simulate Y based on given parameters and regions, and store X and Y values.

    Parameters:
    n (int): Number of simulations
    rho (float): Correlation coefficient between U1 and U2
    beta1 (float): Coefficient for U1
    beta2 (float): Coefficient for U2
    delta1 (float): Threshold adjustment for region01
    delta2 (float): Threshold adjustment for region10
    seed (int, optional): Seed for the random number generator

    Returns:
    tuple: Two numpy arrays, X_vals and Y, both of shape (n, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    # Covariance matrix for the bivariate normal distribution
    cov = [[1, rho], [rho, 1]]

    # Storage for the results
    Y = np.zeros((n, 2))
    X_vals = np.zeros((n, 2))

    # Simulation
    for i in range(n):
        # Generate U from a bivariate normal distribution
        U = np.random.multivariate_normal([0, 0], cov)

        # Generate X from independent Bernoulli distributions
        X = np.random.binomial(1, 0.5, 2)
        X_vals[i] = X

        # Calculate the threshold values for regions
        threshold1_00 = -X[0] * beta1
        threshold2_00 = -X[1] * beta2
        threshold1_01 = -X[0] * beta1 - delta1
        threshold2_10 = -X[1] * beta2 - delta2

        # Determine the region and assign Y
        if U[0] <= threshold1_00 and U[1] <= threshold2_00:
            Y[i] = [0, 0]
        elif U[0] <= threshold1_01 and U[1] >= threshold2_00 and not (U[0] >= threshold1_00 and U[1] <= threshold2_10):
            Y[i] = [0, 1]
        elif U[0] >= threshold1_00 and U[1] <= threshold2_10 and not (U[0] <= threshold1_01 and U[1] >= threshold2_00):
            Y[i] = [1, 0]
        elif U[0] >= threshold1_01 and U[1] >= threshold2_10:
            Y[i] = [1, 1]
        elif (U[0] <= threshold1_01 and U[1] >= threshold2_00) and (U[0] >= threshold1_00 and U[1] <= threshold2_10):
            Y[i] = [0, 1] if np.random.rand() < 0.5 else [1, 0]

    return X_vals, Y

# Example usage
n = 1000
beta1 = 0.75
beta2 = 0.25
delta1 = -0.5
delta2 = -0.5
rho = 0.5
theta_true = [beta1, beta2, delta1, delta2, rho]
seed = 123

# Simulate the values
X, Y = simulate_y_entrygame(n, rho, beta1, beta2, delta1, delta2, seed=seed)
np.savez_compressed('./Data/data_entrygame', X=X, Y=Y)