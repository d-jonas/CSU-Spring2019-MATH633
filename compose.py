"""
This module will generate a song of length N+1 starting from a random chord.
Currently yields error when saving to midi file.

March 6, 2019
"""

import chorales
import train
import torch
import numpy as np

N = 32

chord = np.array([[np.float32(np.random.rand()>0.9) for i in range(88)]])
song = [chord]
pred = torch.tensor(song[0]).view(1,1,88)

network = train.get_network()

for i in range(N):
    pred = network.forward(pred)
    chord = torch.gt(pred,0.05).float().view(1,-1).numpy()
    chord.size
    song.append(chord)
#print(song)
song = np.array(song).reshape(N,88)
print(np.shape(song))
chorales.save_to_midi(song, 'composition.mid', verbosity=1)
