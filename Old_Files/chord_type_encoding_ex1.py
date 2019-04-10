#
# An example demonstrating how to generate 
# one-hot-encoded targets based on either 
#   (a) chord roots (A, A#, B, C, D-, D, etc), or
#   (b) chord type (major chord, minor chord, other).
#
# These could be alternate targets for the LSTM, 
# rather than predicting the following note itself
# (which is 88-dimensiona, so generally expected 
# to be more difficult).
#

import chorales
import time

song = chorales.train[13]

print('')
# Roots for the notes; "A", "A#", "E-" (E flat), etc.
print('Getting roots of all chords in the song...', end='')
time.sleep(0.1)

roots = [chorales.chord_root(note) for note in song.T]
print('done.')

print('First few roots:')
print(roots[:5])

# Types for the notes; "major", "minor", or "other"
print('\nGetting types of all chords in the song...', end='')
time.sleep(0.1)

types = [chorales.chord_type(note) for note in song.T]
print('done.')

print('First few types:')
print(types[:5])


# chorales.py now has dictionaries and a function 
# built-in to one-hot encode these in a consistent manner 
# across all songs (i.e., if a chord root doesn't appear 
# in one song, it won't affect the one-hot encoding in other songs).
# In this manner, the classes are mapped to R^12.
print('')

print('Mapping roots of chords to one-hot-encoded vectors.')
targets = chorales.one_hot_encode(roots)

# The function recognizes the chord type inputs as well and 
# instead maps them to R^3.
print('Mapping types of chords to one-hot-encoded vectors.')
targets2 = chorales.one_hot_encode(types)

