from model import get_decoded_exp
import numpy as np

# URL=https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00264      
# DOI=10.3389/fpsyg.2013.00264
CONSONANCE_ORDER = [
    # "U",
    "O", # Octave
    "P5", # Perfect 5th
    "P4", # Perfect 4th
    "M3", # Major 3rd
    "M6", # Major 6th
    "m6", # Minor 6th
    "m3", # Minor 3rd
    "T", # Augmented 4th / Diminished 5th / Tritone
    "m7", # Minor 7th
    "M2", # Major 2nd
    "M7", # Major 7th
    "m2", # Minor 2nd
]

# From Schwartz et al. 2003
CONS_RANK = [12, 11, 10, 9, 9, 8, 7, 6, 5, 5, 3, 2] # Same order as intervals above


def consonance_field(note):
    intervals = calculate_intervals(note)
    notes = [intervals[interval] for interval in CONSONANCE_ORDER]

    decoded = [get_decoded_exp(note) for note in notes]

    mean = sum(CONS_RANK) / len(CONS_RANK)
    factors = [(rank - mean) for rank in CONS_RANK]

    factored = np.array([factor * note_decoded for factor, note_decoded in zip(factors, decoded)])
    field = np.sum(factored, axis=0) / 12
    return field

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

def calculate_intervals(root_note):
    """Calculate named intervals for a given root note."""
    root_semitone = note_to_semitone(root_note)
    
    interval_to_semitone_map = {
        "U": 0, # Unison
        "m2": 1, # Minor 2nd
        "M2": 2, # Major 2nd
        "m3": 3, # Minor 3rd
        "M3": 4, # Major 3rd
        "P4": 5, # Perfect 4th
        "T": 6, # Augmented 4th / Diminished 5th / Tritone
        "P5": 7, # Perfect 5th
        "m6": 8, # Minor 6th
        "M6": 9, # Major 6th
        "m7": 10, # Minor 7th
        "M7": 11, # Major 7th
        "O": 12 # Octave
    }
    
    intervals = {}
    for interval, semitone_interval in interval_to_semitone_map.items():
        new_semitone = root_semitone + semitone_interval
        intervals[interval] = semitone_to_note(new_semitone)
    
    return intervals

