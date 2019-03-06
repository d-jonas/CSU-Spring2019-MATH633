'''
A second example using the built-in support for 
interpreting chords in chorales.py through the music21 package.
'''

import chorales

chorale = chorales.encoded_train[0]

chords = [chorales.get_chord(c) for c in chorale]

for i,c in enumerate(chords):
    print('Beat %.3i : %s' % (i, c))
