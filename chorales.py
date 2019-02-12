'''
Purpose: load the pickle file containing the Bach chorale data.

This script should be restricted to this focus, so should
only be rarely (if ever) modified.

12 February 2019
Colorado State University

'''

import pickle

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

test = all_data['test']
train = all_data['train']
valid = all_data['valid']
