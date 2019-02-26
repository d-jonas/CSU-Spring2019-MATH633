'''
Example visualization of the data using plain data
chorales.py. Each entry in the chorales.train, .test, .valid
corresponds to a single chorale.

The data is either in "encoded" or plain format.
Encoded just describes locations of nonzero entries.
The plain format has uniform zero-one vectors to be used for LSTM.
'''

import chorales
from matplotlib import pyplot

example = chorales.train[0]

fig,ax = pyplot.subplots(1,1, figsize=(8,5))

ax.imshow( example, cmap=pyplot.cm.Greys )

# Prettify the plot a little. Assume everything is in 4/4 time.
ax.set_xticks( [8*j for j in range(example.shape[1]//8 + 1)] )

# Places notes on y-axis in place of numbers
# Create dictionaries for mapping from midi encoding to notes and back

notes = [
'A0','A0#','B0',
'C1','C1#','D1','D1#','E1','F1','F1#','G1','G1#','A1','A1#','B1',
'C2','C2#','D2','D2#','E2','F2','F2#','G2','G2#','A2','A2#','B2',
'C3','C3#','D3','D3#','E3','F3','F3#','G3','G3#','A3','A3#','B3',
'C4','C4#','D4','D4#','E4','F4','F4#','G4','G4#','A4','A4#','B4',
'C5','C5#','D5','D5#','E5','F5','F5#','G5','G5#','A5','A5#','B5',
'C6','C6#','D6','D6#','E6','F6','F6#','G6','G6#','A6','A6#','B6',
'C7','C7#','D7','D7#','E7','F7','F7#','G7','G7#','A7','A7#','B7',
'C8']

# midi number indexes note
shift=0
midi2note = dict((i+shift, note) for i, note in enumerate(notes))


# note indexes midi number
note2midi = dict((v,k) for k,v in midi2note.items())

skip = 12 # Determines skips between tick marks on y-axis

tone_labels = []

for i in range(shift,shift+88,skip):
    tone_labels.append(midi2note[i])

pyplot.yticks(range(shift,shift+88,skip),tone_labels)

ax.set_xlabel('Beat number', fontsize=16)
ax.set_ylabel('Tone', fontsize=16)
ax.xaxis.grid()
ax.invert_yaxis()

fig.show()
