import numpy as np
from temporal_analysis import calc_avg_isi, get_avg_isi
from spatial_analysis import count_spikes, count_spikes_optimized

# import pytest  # noqa: F401


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

