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


# Define the quadrature data as a multi-line string
# The data points are based on Sparse Grids by Florian Heiss & Viktor Winschel
# http://www.sparse-grids.de/
quadrature_data = """
-4.1849560176727323, -1.7320508075688772, .00001571157625941959
-4.1849560176727323, 0, .00006284630503767847
-4.1849560176727323, 1.7320508075688772, .00001571157625941959
-2.8612795760570582, -1.7320508075688772, .0013327209118155882
-2.8612795760570582, 0, .0053308836472623529
-2.8612795760570582, 1.7320508075688772, .0013327209118155882
-1.7320508075688772, -4.1849560176727323, .00001571157625941958
-1.7320508075688772, -2.8612795760570582, .0013327209118155882
-1.7320508075688772, -1.7320508075688772, .0038392050587172655
-1.7320508075688772, -.74109534999454085, .045012388262989624
-1.7320508075688772, 0, -.0055491031100787513
-1.7320508075688772, .74109534999454085, .045012388262989624
-1.7320508075688772, 1.7320508075688772, .0038392050587172655
-1.7320508075688772, 2.8612795760570582, .0013327209118155882
-1.7320508075688772, 4.1849560176727323, .00001571157625941958
-.74109534999454085, -1.7320508075688772, .04501238826298963
-.74109534999454085, 0, .18004955305195852
-.74109534999454085, 1.7320508075688772, .04501238826298963
0, -4.1849560176727323, .00006284630503767847
0, -2.8612795760570582, .0053308836472623529
0, -1.7320508075688772, -.0055491031100787652
0, -.74109534999454085, .18004955305195852
0, 0, -.1058201058201052
0, .74109534999454085, .18004955305195852
0, 1.7320508075688772, -.0055491031100787652
0, 2.8612795760570582, .0053308836472623529
0, 4.1849560176727323, .00006284630503767847
.74109534999454085, -1.7320508075688772, .04501238826298963
.74109534999454085, 0, .18004955305195852
.74109534999454085, 1.7320508075688772, .04501238826298963
1.7320508075688772, -4.1849560176727323, .00001571157625941958
1.7320508075688772, -2.8612795760570582, .0013327209118155882
1.7320508075688772, -1.7320508075688772, .0038392050587172655
1.7320508075688772, -.74109534999454085, .045012388262989624
1.7320508075688772, 0, -.0055491031100787513
1.7320508075688772, .74109534999454085, .045012388262989624
1.7320508075688772, 1.7320508075688772, .0038392050587172655
1.7320508075688772, 2.8612795760570582, .0013327209118155882
1.7320508075688772, 4.1849560176727323, .00001571157625941958
2.8612795760570582, -1.7320508075688772, .0013327209118155882
2.8612795760570582, 0, .0053308836472623529
2.8612795760570582, 1.7320508075688772, .0013327209118155882
4.1849560176727323, -1.7320508075688772, .00001571157625941959
4.1849560176727323, 0, .00006284630503767847
4.1849560176727323, 1.7320508075688772, .00001571157625941959
"""

# Parse the quadrature data
quadrature_data = np.fromstring(quadrature_data, sep=',').reshape(-1, 3)
quadrature_points = quadrature_data[:, :2]
weights = quadrature_data[:, 2]

def transform_points(quadrature_points, mean, cov):
    chol_cov = cholesky(cov, lower=True)
    transformed_points = np.dot(quadrature_points, chol_cov.T) + mean
    return transformed_points

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

def calculate_Ftheta_panel(X, theta):
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

    # Transform points
    w_points = transform_points(quadrature_points, mean, cov)

    # Calculate probabilities for each region using quadrature points
    probabilities = calculate_probabilities_quadrature(w_points, weights, delta12_x_theta, delta13_x_theta, delta23_x_theta)

    return np.array(probabilities)