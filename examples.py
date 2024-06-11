from scipy.stats import mvn

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