from model import get_decoded_exp
import numpy as np

# URL=https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00264
# DOI=10.3389/fpsyg.2013.00264
CONSONANCE_ORDER = [
    # "U",
    "O",  # Octave
    "P5",  # Perfect 5th
    "P4",  # Perfect 4th
    "M3",  # Major 3rd
    "M6",  # Major 6th
    "m6",  # Minor 6th
    "m3",  # Minor 3rd
    "T",  # Augmented 4th / Diminished 5th / Tritone
    "m7",  # Minor 7th
    "M2",  # Major 2nd
    "M7",  # Major 7th
    "m2",  # Minor 2nd
]

# From Schwartz et al. 2003
CONS_RANK = [2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 11, 12]  # Same order as intervals above
# Lower is more consonant


def note_to_semitone(note):
    """Convert a note to its corresponding semitone value relative to C0."""
    note_to_semitone_map = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
        'E': 4, 'F': 5, 'F#': 6, 'G': 7,
        'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    note_name = note[:-1]
    octave = int(note[-1])
    return note_to_semitone_map[note_name] + octave * 12


def semitone_to_note(semitone):
    """Convert a semitone value to its corresponding note name relative to C0."""
    semitone_to_note_map = [
        'C', 'C#', 'D', 'D#', 'E', 'F',
        'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    note_name = semitone_to_note_map[semitone % 12]
    octave = semitone // 12
    return f"{note_name}{octave}"


def consonance_probabilities(ranks=CONS_RANK):
    # Inverse the ranks
    inverse_values = [13 - rank for rank in ranks]

    # Normalize the values
    total = sum(inverse_values)
    return [val/total for val in inverse_values]


def empirical_probabilities():
    return consonance_probabilities()


def consonance_scores(ranks=CONS_RANK, min_value=-1, max_value=1, adjust_to_zero=False):
    """
    Return the consonance scores for the given consonance ranks.
    Args:
        ranks (_type_, optional): Consonance Ranks for the intervals. Defaults to CONS_RANK which is based off empirical data.
        min_value (int, optional): Minimum possible probability. Defaults to 0.01.
        max_value (int, optional): Maximum possible probability. Defaults to 1.
        adjust_to_zero (bool, optional): Make the sum of scores 0. Defaults to False.

    Returns:
        _type_: np.array(dtype=float)
    """
    # Constants
    min_rank = 1
    max_rank = 12

    # 1. Invert the rank values
    inverted_ranks = [13 - rank for rank in ranks]

    # 2. Normalize to [0, 1]
    normalized_values = [(inv_rank - min_rank) / (max_rank - min_rank) for inv_rank in inverted_ranks]

    # 3. Rescale based on the given min_value and max_value
    range_diff = max_value - min_value
    rescaled_values = [min_value + (range_diff * norm_val) for norm_val in normalized_values]

    if adjust_to_zero:
        # 4. Adjust values to ensure their sum is 0
        avg_value = sum(rescaled_values) / len(rescaled_values)
        adjusted_values = np.array([val - avg_value for val in rescaled_values])

        return np.round(adjusted_values, 4)

    return np.round(rescaled_values, 4)


def consonance_ordered_notes(root_note):
    """Return the consonance order of the notes in the scale of the given root note."""
    root_semitone = note_to_semitone(root_note)

    interval_to_semitone_map = {
        "U": 0,  # Unison
        "m2": 1,  # Minor 2nd
        "M2": 2,  # Major 2nd
        "m3": 3,  # Minor 3rd
        "M3": 4,  # Major 3rd
        "P4": 5,  # Perfect 4th
        "T": 6,  # Augmented 4th / Diminished 5th / Tritone
        "P5": 7,  # Perfect 5th
        "m6": 8,  # Minor 6th
        "M6": 9,  # Major 6th
        "m7": 10,  # Minor 7th
        "M7": 11,  # Major 7th
        "O": 12  # Octave
    }

    intervals_map = {}
    for interval, semitone_interval in interval_to_semitone_map.items():
        new_semitone = root_semitone + semitone_interval
        intervals_map[interval] = semitone_to_note(new_semitone)

    return [intervals_map[interval] for interval in CONSONANCE_ORDER]
