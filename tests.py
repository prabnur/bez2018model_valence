from unittest import mock
import numpy as np
from analysis.probability.simple import generate_probabilities_simple
from analysis.temporal import calc_avg_isi, get_avg_isi
from analysis.spatial import count_spikes, count_spikes_optimized
from analysis.spike_tensor import generate_snapshot, generate_spike_tensor, generate_expectation
from analysis.musical import note_to_semitone, semitone_to_note

from evaluate import predicted_consonance_scores, predicted_probabilities
from testhelpers import array_equal


def test_generate_probabilities_simple():
    mockTensors = np.array([
        [[[0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 1]]],
        [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 0]]],
        [[[1, 1, 1], [0, 0, 0]], [[1, 0, 1], [1, 0, 0]]],
        [[[0, 1, 0], [0, 1, 1]], [[0, 0, 1], [1, 1, 1]]],
        [[[0, 1, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 0]]],
        [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 1]]],
        [[[0, 1, 0], [0, 1, 0]], [[1, 0, 1], [0, 0, 1]]],
        [[[0, 0, 1], [1, 0, 1]], [[1, 0, 0], [0, 1, 1]]],
        [[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]]],
        [[[1, 1, 1], [0, 0, 1]], [[0, 1, 0], [1, 1, 0]]]
    ])
    output = generate_probabilities_simple("", mockTensors=mockTensors)
    expected_output = np.array(
        [[[.2, .6, .4], [.3, .3, .4]], [[.5, .3, .4], [.6, .5, .6]]]
    )
    array_equal(output, expected_output, roundFactor=1)


def test_evaluate():
    # predicted_consonance_scores
    mock_tensor = np.array([
        [[1, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 1]]
    ])

    probability_tensor = np.array([
        [[.24, .01, .69], [.84, .38, .52]],
        [[.11, .91, .36], [.28, .55, .77]]
    ])

    tensors = np.array([mock_tensor] * 12)

    expected_output = np.array([0.5] * 12)
    output = predicted_consonance_scores(probability_tensor, tensors, mock_tensor)
    assert array_equal(output, expected_output, roundFactor=1)

    # predicted_probabilities
    scores = [0.9145, 0.1476, 0.6272, 0.4559]
    expected_output = [0.43, 0.07, 0.29, 0.21]
    assert array_equal(predicted_probabilities(scores), expected_output, roundFactor=2)
    assert sum(predicted_probabilities(scores)) == 1


def test_temporal_analyis():
    # calc_avg_isi
    assert calc_avg_isi(np.array([1, 2, 3, 0, 0, 0])) == 1
    assert calc_avg_isi(np.array([2, 4, 0, 0, 0, 0])) == 2
    assert calc_avg_isi(np.array([0, 3, 6, 9, 0, 0])) == 3

    # get_avg_isi
    test_data = np.array(
        [
            [[1, 2, 3, 0, 0, 0], [2, 4, 6, 0, 0, 0], [3, 6, 9, 0, 0, 0]],
            [[3, 6, 9, 0, 0, 0], [2, 4, 6, 0, 0, 0], [1, 2, 3, 0, 0, 0]],
        ]
    )
    assert np.array_equal(
        get_avg_isi(test_data), np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    )


def test_spatial_analysis():
    test_data = np.array(
        [
            [[1, 2, 3, 4, 0, 0], [2, 4, 6, 4, 5, 0], [3, 6, 0, 0, 0, 0]],
            [[3, 0, 0, 0, 0, 0], [2, 4, 6, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        ]
    )
    assert count_spikes(test_data) == [4, 5, 2, 1, 3, 0]
    assert np.array_equal(
        count_spikes_optimized(test_data), np.array([4, 5, 2, 1, 3, 0])
    )


def test_note_to_semitone():
    assert note_to_semitone('C0') == 0
    assert note_to_semitone('C#0') == 1
    assert note_to_semitone('D0') == 2
    assert note_to_semitone('E0') == 4
    assert note_to_semitone('A2') == 33
    assert note_to_semitone('C4') == 48


def test_semitone_to_note():
    assert semitone_to_note(0) == 'C0'
    assert semitone_to_note(1) == 'C#0'
    assert semitone_to_note(2) == 'D0'
    assert semitone_to_note(4) == 'E0'
    assert semitone_to_note(33) == 'A2'
    assert semitone_to_note(48) == 'C4'


# def test_calculate_intervals():
#     expected_intervals_C0 = {
#         "U": "C0", "m2": "C#0", "M2": "D0", "m3": "D#0", "M3": "E0",
#         "P4": "F0", "T": "F#0", "P5": "G0", "m6": "G#0", "M6": "A0",
#         "m7": "A#0", "M7": "B0", "O": "C1"
#     }
#     assert calculate_intervals('C0') == expected_intervals_C0

#     expected_intervals_A2 = {
#         "U": "A2", "m2": "A#2", "M2": "B2", "m3": "C3", "M3": "C#3",
#         "P4": "D3", "T": "D#3", "P5": "E3", "m6": "F3", "M6": "F#3",
#         "m7": "G3", "M7": "G#3", "O": "A3"
#     }
#     assert calculate_intervals('A2') == expected_intervals_A2


def test_spike_tensor():
    # TENSOR Generation
    spikes = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9], [0.7, 0.3, 0]],
    ])
    expected_output = np.array([
        [[1, 1, 1, 0, 0, 0.], [0, 0, 0, 1, 1, 1.]],
        [[0, 0, 0, 0, 0, 0.], [0, 0, 1, 0, 0, 0.]]
    ])
    output = generate_spike_tensor(spikes, tau=0.1, duration=0.6)
    assert np.array_equal(output, expected_output)

    # SNAPSHOT
    tensor = [[[0, 1, 0, 1], [1, 0, 0, 1]]]
    expectation = [[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]
    output = generate_snapshot(tensor, expectation, snap_size=3)
    expected_output = np.array([0.367, 0.475, 0.45])
    assert array_equal(output, expected_output, roundFactor=3)

    tensor = [[[0, 0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0]]]
    expectation = [[[0, 0, 0.1, 0.2, 0.3, 0.4, 0, 0, 0], [0.5, 0.6, 0.7, 0.8, 0, 0, 0, 0, 0]]]
    output = generate_snapshot(tensor, expectation, snap_size=3)
    expected_output = np.array([0.367, 0.475, 0.45])
    assert array_equal(output, expected_output, roundFactor=3)

    tensor = [[[0, 0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0]]]
    expectation = [[[0, 0, 0.1, 0.2, 0.3, 0.4, 0, 0, 0], [0.5, 0.6, 0.7, 0.8, 0, 0, 0, 0, 0]]]
    output = np.round(generate_snapshot(tensor, expectation, snap_size=5), 3)
    assert output[1] == 0.367
    assert output[2] == 0.475
    assert output[3] == 0.45

    # EXPECTATION
    tensors = np.array([
        [[[0, 0, 0, 1, 1, 1],
          [0, 0, 0, 0, 1, 1]],
         [[1, 0, 1, 0, 0, 1],
          [1, 0, 1, 0, 0, 1]]],

        [[[1, 0, 1, 1, 1, 0],
          [1, 0, 1, 0, 0, 0]],
         [[0, 0, 1, 1, 0, 1],
          [1, 1, 1, 1, 1, 0]]]
    ])
    scores = np.array([0.2, 0.4])
    expected_output = np.array([
        [[0.4, 0., 0.4, 0.3, 0.3, 0.2],
         [0.4, 0., 0.4, 0., 0.2, 0.2]],
        [[0.2, 0., 0.3, 0.4, 0., 0.3],
         [0.3, 0.4, 0.3, 0.4, 0.4, 0.2]]
    ])
    output = generate_expectation(tensors, scores)
    assert array_equal(output, expected_output, roundFactor=1)


if __name__ == "__main__":
    print("Usage: pytest tests.py")
