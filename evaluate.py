from scipy.optimize import linprog
from analysis.musical import consonance_ordered_notes, empirical_probabilities
from analysis.pulse import MAX, TAU
from analysis.spike_tensor import generate_spike_tensor

import numpy as np

from cache import get_spikes


def predicted_consonance_scores(probability_tensor, consonance_ordered_tensors, root_tensor):
    scores = np.array([])

    for tensor in consonance_ordered_tensors:
        product = tensor * probability_tensor
        scores = np.append(scores, product.sum())

    # TODO: Think more about this
    max_possible = np.count_nonzero(root_tensor)

    return scores / max_possible


def predicted_probabilities(scores):
    total = sum(scores)
    return [score / total for score in scores]


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


def evaluate_single(probability_tensor, root_note="C4", debug=False):
    notes = consonance_ordered_notes(root_note)
    notes_spikes = [get_spikes(note) for note in notes]
    consonance_ordered_tensors = [generate_spike_tensor(spikes) for spikes in notes_spikes]

    root_tensor = generate_spike_tensor(get_spikes(root_note))

    scores = predicted_consonance_scores(probability_tensor, consonance_ordered_tensors, root_tensor)

    if debug:
        print("Scores")
        print(scores)
        print("P_predicted")
        print(P_predicted)
        print("P_empirical")
        print(empirical_probabilities())

    P_predicted = predicted_probabilities(scores)
    return earth_movers_distance(P_predicted)
