"""
This module will generate a song of length N+1 starting from a random chord.
Currently yields error when saving to midi file.

March 6, 2019
"""

import chorales
import train
import torch
import numpy as np
from train import loss_library

N = 32

chord = np.array([[np.float32(np.random.rand()>0.9) for i in range(88)]])
song = [chord]
pred = torch.tensor(song[0]).view(1,1,88)

# Init and train network
network = train.get_network()
optimizer = torch.optim.SGD(network.parameters(), lr = 0.5, momentum = 0.5)
data = chorales.train
network, losses = train.train(network, loss_library['MSELoss'], optimizer, data, epochs = 5)

#
for i in range(N):
    pred = network.forward(pred)
    chord = torch.gt(pred,0.05).float().view(1,-1).numpy()
    chord.size
    song.append(chord)


#print(song)
song = np.array(song).reshape(88,N+1)
chorales.save_to_midi(song, 'composition.mid', verbosity=1)
