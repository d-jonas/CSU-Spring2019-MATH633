'''
Purpose: load the pickle file containing the Bach chorale data.

This script should be restricted to this focus, so should
only be rarely (if ever) modified.

12 February 2019
Colorado State University

'''

import pickle
import numpy as np

# Assume the data lies in the same folder as this
# script. If it doesn't, modify this string to
# indicate the correct path.
prefix = './'

with open(prefix + 'JSB Chorales.pickle','rb') as f:
    all_data = pickle.load(f)


#
# Data is structured as a dictionary broken
# in to training, testing and validation data.
# Unpack into separate variables so that
# the data can be called from import by
# doing, e.g. chorales.train[0].
#

encoded_test = all_data['test']
encoded_train = all_data['train']
encoded_valid = all_data['valid']

test = []
for encoded_chorale in encoded_test:
    chorale = np.zeros( (88,len(encoded_chorale)), dtype=float)

    for i,beat in enumerate(encoded_chorale):
        # According to Patrick, notes start at 21, so shift down.
        beat_shft = np.array(beat, dtype=int) - 21
        chorale[beat_shft,i] = 1.
    #
    test.append( chorale )
#

train = []
for encoded_chorale in encoded_train:
    chorale = np.zeros( (88,len(encoded_chorale)), dtype=float)

    for i,beat in enumerate(encoded_chorale):
        # According to Patrick, notes start at 21, so shift down.
        beat_shft = np.array(beat, dtype=int) - 21
        chorale[beat_shft,i] = 1.
    #
    train.append( chorale )
#

valid = []
for encoded_chorale in encoded_valid:
    chorale = np.zeros( (88,len(encoded_chorale)), dtype=float)

    for i,beat in enumerate(encoded_chorale):
        # According to Patrick, notes start at 21, so shift down.
        beat_shft = np.array(beat, dtype=int) - 21
        chorale[beat_shft,i] = 1.
    #
    valid.append( chorale )
#
