from analysis.musical import consonance_ordered_notes, empirical_probabilities
from analysis.spike_tensor import generate_spike_tensor
from scipy.spatial.distance import jensenshannon

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
    allPositive = all(np.array(scores) >= 0)
    if allPositive:
        total = sum(scores)
        return [score / total for score in scores]

    # Normalize numbers to be positive
    min_num = min(scores)
    if min_num < 0:
        scores = [num - min_num for num in scores]

    # Normalize scores to sum to 1
    total_weight = sum(scores)
    probabilities = [weight / total_weight for weight in scores]

    return probabilities


def js_divergence(P_predicted):
    P_empirical = np.array(empirical_probabilities())
    return jensenshannon(P_predicted, P_empirical)


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
    return round(100 * (1 - js_divergence(P_predicted)))
