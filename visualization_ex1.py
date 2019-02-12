'''
Example visualization of the data using encoded data
chorales.py. Each entry in the chorales.train, .test, .valid
corresponds to a single chorale.

The data is either in "encoded" or plain format.
Encoded just describes locations of nonzero entries.
The plain format has uniform zero-one vectors to be used for LSTM.
'''

import chorales
from matplotlib import pyplot

example = chorales.encoded_train[0]

fig,ax = pyplot.subplots(1,1, figsize=(12,5))

for i,beat in enumerate(example):
    ax.scatter( [i+1 for _ in beat] , beat, s=10, c='k')
#

# Prettify the plot a little. Assume everything is in 4/4 time.
ax.set_xticks( [8*j for j in range((i+1)//8 + 1)] )

# TODO: associate the numerical values with usual tones
# (e.g. what does a tone of "80" correspond to on the keyboard?)

ax.set_xlabel('Beat number', fontsize=16)
ax.set_ylabel('Tone', fontsize=16)
ax.xaxis.grid()

fig.show()