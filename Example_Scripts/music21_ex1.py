import music21
import chorales
import numpy as np

def print_chord(thing):
    print(thing,' is a ', thing.pitchedCommonName, 'chord.')
#

# relevant submodule
ch = music21.chord

############################

# get the encoded locations of a chord
ex_loc = np.where( chorales.train[0][:,0] )[0]

#ex = ch.Chord(ex_loc) # this throws an exception -- why???

# for the moment, manually enter the chord.
ex = ch.Chord([39,51,58,67])

# Wow!
print(ex.pitchedCommonName)
print_chord(ex)

# Another example - C7 chord; C-E-G-Bflat
ex2 = ch.Chord([60,64,67,70])
print_chord(ex2)

# Note changing to a B changes the name - Cmaj7
ex3 = ch.Chord([60,64,67,71])
print_chord(ex3)
