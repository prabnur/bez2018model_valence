import numpy as np
from analysis.musical import empirical_probabilities
from scipy.optimize import linprog


def earth_movers_distance(P_predicted):
    P_empirical = empirical_probabilities()
    if len(P_predicted) != len(P_empirical):
        raise ValueError("Both lists must have the same number of elements")

    n = len(P_predicted)
    dist_matrix = np.zeros((n, n))

    # Creating distance matrix
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = abs(i - j)  # simple 1D distance

    # Flatten the distance matrix to create the cost vector
    c = dist_matrix.reshape(n * n, )

    # Constraints
    # Each row of A indicates how much of the probability from a given point i is moved to other points
    A = np.zeros((n, n * n))
    for i in range(n):
        for j in range(n):
            A[i, i * n + j] = 1
            A[i, j * n + i] = -1

    # Difference between predicted and empirical probabilities
    b_eq = np.array(P_predicted) - np.array(P_empirical)

    # Linear programming
    result = linprog(c, A_eq=A, b_eq=b_eq, bounds=(0, None), method='highs')

    if not result.success:
        raise ValueError("Linear programming failed")

    distance = result.fun
    max_distance = 11.0
    return 1 - (distance / max_distance)


def kl_divergence(P_predicted):
    """
    Calculate the Kullback-Leibler Divergence from distribution A to B.
    A and B are lists of probabilities and must be of the same length.
    """
    P_predicted = np.array(P_predicted)
    P_empirical = np.array(empirical_probabilities())

    # Replace 0s with very small numbers to avoid division by zero
    P_empirical = np.where(P_empirical == 0, 1e-10, P_empirical)

    return np.sum(P_predicted * np.log(P_predicted / P_empirical))


def kl_divergence_reverse(P_predicted):
    """
    Calculate the Kullback-Leibler Divergence from distribution A to B.
    A and B are lists of probabilities and must be of the same length.
    """
    P_predicted = np.array(P_predicted)
    P_empirical = np.array(empirical_probabilities())

    Temp = P_predicted
    P_predicted = P_empirical
    P_empirical = Temp

    # Replace 0s with very small numbers to avoid division by zero
    P_empirical = np.where(P_empirical == 0, 1e-10, P_empirical)

    return np.sum(P_predicted * np.log(P_predicted / P_empirical))
