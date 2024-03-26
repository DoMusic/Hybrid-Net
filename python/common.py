CHORD_LABELS = [
    'N', 'X',
    # maj
    'C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
    # min
    'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
    # 7
    'C:7', 'C#:7', 'D:7', 'D#:7', 'E:7', 'F:7', 'F#:7', 'G:7', 'G#:7', 'A:7', 'A#:7', 'B:7',
    # maj7
    'C:maj7', 'C#:maj7', 'D:maj7', 'D#:maj7', 'E:maj7', 'F:maj7', 'F#:maj7', 'G:maj7', 'G#:maj7', 'A:maj7', 'A#:maj7',
    'B:maj7',
    # min7
    'C:min7', 'C#:min7', 'D:min7', 'D#:min7', 'E:min7', 'F:min7', 'F#:min7', 'G:min7', 'G#:min7', 'A:min7', 'A#:min7',
    'B:min7',
    # 6
    'C:6', 'C#:6', 'D:6', 'D#:6', 'E:6', 'F:6', 'F#:6', 'G:6', 'G#:6', 'A:6', 'A#:6', 'B:6',
    # m6
    'C:m6', 'C#:m6', 'D:m6', 'D#:m6', 'E:m6', 'F:m6', 'F#:m6', 'G:m6', 'G#:m6', 'A:m6', 'A#:m6', 'B:m6',
    # sus2
    'C:sus2', 'C#:sus2', 'D:sus2', 'D#:sus2', 'E:sus2', 'F:sus2', 'F#:sus2', 'G:sus2', 'G#:sus2', 'A:sus2', 'A#:sus2',
    'B:sus2',
    # sus4
    'C:sus4', 'C#:sus4', 'D:sus4', 'D#:sus4', 'E:sus4', 'F:sus4', 'F#:sus4', 'G:sus4', 'G#:sus4', 'A:sus4', 'A#:sus4',
    'B:sus4',
    # 5
    'C:5', 'C#:5', 'D:5', 'D#:5', 'E:5', 'F:5', 'F#:5', 'G:5', 'G#:5', 'A:5', 'A#:5', 'B:5',
]

SEGMENT_LABELS = [
    'start',
    'end',
    'intro',
    'outro',
    'verse',
    'chorus',
    'solo',
    'break',
    'bridge',
    'inst',
]

MAJOR_CHORDS = [
    ["C", "Dm", "Em", "F", "G", "Am", "G7"],
    ["G", "Am", "Bm", "C", "D", "Em", "D7"],
    ["D", "Em", "F#m", "G", "A", "Bm", "A7"],
    ["A", "Bm", "C#m", "D", "E", "F#m", "E7"],
    ["E", "F#m", "G#m", "A", "B", "C#m", "B7"],
    ["F", "Gm", "Am", "Bb", "C", "Dm", "C7"],
    ["B", "C#m", "D#m", "E", "F#", "G#m", "F#7"],
    ["Db", "Ebm", "Fm", "Gb", "Ab", "Bbm", "Ab7"],
    ["Eb", "Fm", "Gm", "Ab", "Bb", "Cm", "Bb7"],
    ["Gb", "Abm", "Bbm", "Cb", "Db", "Ebm", "Db7"],
    ["Ab", "Bbm", "Cm", "Db", "Eb", "Fm", "Eb7"],
    ["Bb", "Cm", "Dm", "Eb", "F", "Gm", "F7"],
    ["C#", "D#m", "E#m", "F#", "G#", "A#m", "G#7"],
    ["Cb", "Dbm", "Ebm", "Fb", "Gb", "Abm", "Gb7"],
    ["F#", "G#m", "A#m", "B", "C#", "D#m", "C#7", ],
]

MINOR_CHORDS = [
    ["Am", "Bdim", "C", "Dm", "Em", "F", "G"],
    ["Em", "F#dim", "G", "Am", "Bm", "C", "D"],
    ["Bm", "C#dim", "D", "Em", "F#m", "G", "A"],
    ["F#m", "G#dim", "A", "Bm", "C#m", "D", "E"],
    ["C#m", "D#dim", "E", "F#m", "G#m", "A", "B"],
    ["Dm", "Edim", "F", "Gm", "Am", "Bb", "C"],
    ["G#m", "A#dim", "B", "C#m", "D#m", "E", "F#"],
    ["Bbm", "Cdim", "Db", "Ebm", "Fm", "Gb", "Ab"],
    ["Cm", "Ddim", "Eb", "Fm", "Gm", "Ab", "Bb"],
    ["Ebm", "Fdim", "Gb", "Abm", "Bbm", "Cb", "Db"],
    ["Fm", "Gdim", "Ab", "Bbm", "Cm", "Db", "Eb"],
    ["Gm", "Adim", "Bb", "Cm", "Dm", "Eb", "F"],
    ["A#m", "B#dim", "C#", "D#m", "E#m", "F#", "G#"],
    ["Abm", "Bbdim", "Cb", "Dbm", "Ebm", "Fb", "Gb"],
    ["D#m", "E#dim", "F#", "G#m", "A#m", "B", "C#"],
]

MAJ2MIN_MAP = {
    "G": "Em",
    "C": "Am",
    "D": "Bm",
    "E": "C#m",
    "F": "Dm",
    "A": "F#m",
    "B": "G#m",
    "Db": "Bbm",
    "Eb": "Cm",
    "Gb": "Ebm",
    "Ab": "Fm",
    "Bb": "Gm",
    "C#": "A#m",
    "Cb": "Abm",
    "F#": "D#m"
}
MIN2MAJ_MAP = {v: k for k, v in MAJ2MIN_MAP.items()}
