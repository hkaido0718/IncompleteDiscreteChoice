import numpy as np
#from scipy.stats import mvn
from scipy.stats import multivariate_normal 
from numba import njit, prange

def calculate_Ftheta_entrygame(X, theta):
    
    # Unpack theta
    beta1, beta2, delta1, delta2, rho = theta
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]  # Covariance matrix for U1, U2

    def mvn_prob(lower, upper):
        """ Compute the probability for the bivariate normal distribution. """
        lower = np.array(lower)
        upper = np.array(upper)
        # p, _ = mvn.mvnun(lower, upper, mean, cov)  # compute MVN prob of a rectangle
        p = multivariate_normal(mean=mean, cov=cov).cdf(upper, lower_limit=lower) # compute MVN prob of a rectangle
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

@njit(parallel=True)
def calculate_probabilities(samples, delta12_x_theta, delta13_x_theta, delta23_x_theta):
    n = samples.shape[0]
    prob_region1 = 0
    prob_region2 = 0
    prob_region3 = 0
    prob_region4 = 0
    prob_region5 = 0
    prob_region6 = 0

    for i in prange(n):
        x0, x1 = samples[i]
        if x0 < -delta12_x_theta and x1 < x0 - delta23_x_theta:
            prob_region1 += 1
        elif x0 > -delta12_x_theta and x1 < -delta13_x_theta:
            prob_region2 += 1
        elif x1 > -delta13_x_theta and x1 < x0 - delta23_x_theta:
            prob_region3 += 1
        elif x0 > -delta12_x_theta and x1 > x0 - delta23_x_theta:
            prob_region4 += 1
        elif x0 < -delta12_x_theta and x1 > -delta13_x_theta:
            prob_region5 += 1
        elif x1 < -delta13_x_theta and x1 > x0 - delta23_x_theta:
            prob_region6 += 1

    return (
        prob_region1 / n,
        prob_region2 / n,
        prob_region3 / n,
        prob_region4 / n,
        prob_region5 / n,
        prob_region6 / n
    )

def calculate_Ftheta_panel(X, theta, num_samples=10000, random_seed=123):
    np.random.seed(random_seed)

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
    mean = [0, 0]
    cov = [[2, 1], [1, 2]]  # Covariance matrix for (D12U, D13U)

    # Sample points using NumPy multivariate normal
    samples = np.random.multivariate_normal(mean, cov, size=num_samples)

    # Calculate probabilities for each region using Numba
    probabilities = calculate_probabilities(samples, delta12_x_theta, delta13_x_theta, delta23_x_theta)

    return np.array(probabilities)