"""
This module will generate a song of length N+1 starting from a random chord
given a network that has already been trained.

March 12, 2019
"""

import chorales
import train
import torch
import numpy as np
from train import loss_library

N = 32
data = chorales.train

# Init random chord that net has seen before
rand_song = data[np.random.randint(0,len(data))]
rand_chord = rand_song[:,np.random.randint(0,rand_song.shape[1] - 1)]


song = [rand_chord]
pred = torch.tensor(song[0]).view(1,1,88).float()

# Init and train network
network = train.get_network()
optimizer = torch.optim.SGD(network.parameters(), lr = 0.05, momentum = 0.5)
network, losses = train.train(network, loss_library['MSELoss'], optimizer, data, epochs = 2)


# Lara's version (picking top 4 values and setting them to 1, else set to 0)
for i in range(N):
    pred = network.forward(pred)
    prednp = pred.detach().numpy().reshape(88,)
    chord = train.get_4notes(prednp)
    song.append(chord)

# Codie's version (setting all values above 0.05 to 1, else set to 0)
# for i in range(N):
#     pred = network.forward(pred)
#     chord = torch.gt(pred, 0.05).float().view(1,-1).numpy()
#     chord.size
#     song.append(chord)


song = np.array(song).reshape(88,N+1)
# chorales.save_to_midi(song, 'composition.mid', verbosity=1)
