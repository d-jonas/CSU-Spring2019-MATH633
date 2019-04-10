'''
Demonstrating how you can save the data we're working with to a midi.
'''

import chorales
import numpy as np

# saving the sequenes of 88 column vectors.
# note that this function does the shifting of the notes for you.
chorales.save_to_midi( chorales.train[42], 'chorale_train42.mid', verbosity=1)

# you can also do this with encoded stuff.
chorales.save_to_midi( chorales.encoded_train[1], 'chorale_train01.mid', verbosity=1)

# random song!
song = np.random.rand(88,32)>0.9

chorales.save_to_midi(song, 'random.mid', verbosity=1)
