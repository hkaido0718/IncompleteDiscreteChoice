import numpy as np
from scipy.stats import mvn
from scipy.stats import multivariate_normal
from numba import njit, prange
from scipy.linalg import cholesky
from scipy.stats import qmc, norm

def calculate_Ftheta_entrygame(X, theta):
    
    # Unpack theta
    beta1, beta2, delta1, delta2, rho = theta
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]  # Covariance matrix for U1, U2

    def mvn_prob(lower, upper):
        """ Compute the probability for the bivariate normal distribution. """
        lower = np.array(lower)
        upper = np.array(upper)
        p, _ = mvn.mvnun(lower, upper, mean, cov)  # compute MVN prob of a rectangle
        return p

    # Extract X values
    X1, X2 = X

    # Define the boundaries for each region
    lower_region00 = [-np.inf, -np.inf]
    upper_region00 = [-X1 * beta1, -X2 * beta2]

    lower_region01 = [-np.inf, -X2 * beta2]
    upper_region01 = [-X1 * beta1 - delta1, np.inf]

    lower_region10 = [-X1 * beta1, -np.inf]
    upper_region10 = [np.inf, -X2 * beta2 - delta2]

    lower_region11 = [-X1 * beta1 - delta1, -X2 * beta2 - delta2]
    upper_region11 = [np.inf, np.inf]

    # Calculate probabilities for each region using the CDF of the multivariate normal
    prob_region00 = mvn_prob(lower_region00, upper_region00)
    prob_region01 = mvn_prob(lower_region01, upper_region01)
    prob_region10 = mvn_prob(lower_region10, upper_region10)
    prob_region11 = mvn_prob(lower_region11, upper_region11)

    # Calculate the intersection and difference probabilities
    lower_intersection = [-X1 * beta1, -X2 * beta2]
    upper_intersection = [-X1 * beta1 - delta1, -X2 * beta2 - delta2]
    prob_intersection = mvn_prob(lower_intersection, upper_intersection)

    prob_a = prob_region00
    prob_b = prob_region01 - prob_intersection
    prob_c = prob_region10 - prob_intersection
    prob_d = prob_region11
    prob_e = prob_intersection

    # Return the probabilities as a 5-dimensional array
    return np.array([prob_a, prob_b, prob_c, prob_d, prob_e])


def quasi_monte_carlo_quadrature(n_points, mean, cov):
    # Use Sobol sequence for quasi-Monte Carlo
    sobol = qmc.Sobol(d=2, scramble=True)
    z_points = sobol.random(n_points)

    # Convert Sobol points to standard normal using inverse CDF (ppf)
    z_points = norm.ppf(z_points)
    
    # Transform the points using the covariance matrix
    chol_cov = cholesky(cov, lower=True)
    w_points = np.dot(chol_cov, z_points.T).T + mean

    return w_points, np.full(n_points, 1/n_points)  # equal weights

def calculate_probabilities_quadrature(w_points, weights, delta12_x_theta, delta13_x_theta, delta23_x_theta):
    prob_region1 = 0
    prob_region2 = 0
    prob_region3 = 0
    prob_region4 = 0
    prob_region5 = 0
    prob_region6 = 0

    for i in range(w_points.shape[0]):
        x0, x1 = w_points[i]
        weight = weights[i]

        if x0 < -delta12_x_theta and x1 < x0 - delta23_x_theta:
            prob_region1 += weight
        elif x0 > -delta12_x_theta and x1 < -delta13_x_theta:
            prob_region2 += weight
        elif x1 > -delta13_x_theta and x1 < x0 - delta23_x_theta:
            prob_region3 += weight
        elif x0 > -delta12_x_theta and x1 > x0 - delta23_x_theta:
            prob_region4 += weight
        elif x0 < -delta12_x_theta and x1 > -delta13_x_theta:
            prob_region5 += weight
        elif x1 < -delta13_x_theta and x1 > x0 - delta23_x_theta:
            prob_region6 += weight

    return (
        prob_region1,
        prob_region2,
        prob_region3,
        prob_region4,
        prob_region5,
        prob_region6
    )

def calculate_Ftheta_panel(X, theta, n_points=1024):
    T = 3
    d = len(theta)  # Dimensionality of theta (should be 2 in this case)

    # Extract X values
    X1 = X[0:2]
    X2 = X[2:4]
    X3 = X[4:6]

    # Calculate the dot products
    x1theta = np.dot(X1, theta)
    x2theta = np.dot(X2, theta)
    x3theta = np.dot(X3, theta)

    # Calculate thresholds
    delta12_x_theta = x1theta - x2theta
    delta13_x_theta = x1theta - x3theta
    delta23_x_theta = x2theta - x3theta

    # Define the mean and covariance matrix for the bivariate normal distribution
    mean = np.zeros(2)
    cov = np.array([[2, 1], [1, 2]])  # Covariance matrix for (D12U, D13U)

    # Get quasi-Monte Carlo points and weights
    w_points, weights = quasi_monte_carlo_quadrature(n_points, mean, cov)

    # Calculate probabilities for each region using quadrature points
    probabilities = calculate_probabilities_quadrature(w_points, weights, delta12_x_theta, delta13_x_theta, delta23_x_theta)

    return np.array(probabilities)