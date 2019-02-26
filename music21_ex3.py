'''
A third example, dumping the beats, notes, and 
inferred chords from music21 into a csv file 
for the first chorale in the training set.
'''

import music21
import chorales
import re
import numpy as np

chorale = chorales.encoded_train[0]

# List of chords with detailed information
chs = [music21.chord.Chord([int(t) for t in beat]) for beat in chorale]

# low-tech solution to get the names of notes in the chord.
pattern = '.*Chord ([ABCDEFG0-9\ \-\#]{1,})\>'

table = []
for i,c in enumerate(chs):
    beat = i
    rematch = re.match(pattern, str(c))
    notes = rematch.groups(0)[0]
    chord = c.pitchedCommonName

    table.append([beat, notes, chord])
#

table = [['beat','notes','chord']] + table

np.savetxt('chorale_notes_chords.csv', table, fmt='%s', delimiter=',')
