"""
This module will generate a song of length N+1 starting from a random chord.

song_np is a list of numpy arrays of size (88,)
song_torch is a 3D torch tensor of size (?,?,?)

March 7, 2019
"""

import chorales
import train
import torch
import numpy as np
from train import loss_library

N = 50
data = chorales.train

# Init random chord that net has seen before
rand_song = data[np.random.randint(0,len(data))]
rand_chord = rand_song[:,np.random.randint(0,rand_song.shape[1] - 1)]


song_np = rand_chord
song_torch = torch.tensor(rand_chord).view(1,1,88).float()

# Init and train network
network = train.get_network()
optimizer = torch.optim.SGD(network.parameters(), lr = 0.05, momentum = 0.5)
network, losses = train.train(network, loss_library['MSELoss'], optimizer, data, epochs = 50)

for i in range(N-1):
    network.hidden = network.init_hidden(minibatch_size = i + 1)
    pred_chord_torch = network.forward(song_torch)[:,-1,:]
    next_chord_np = train.get_4_notes(pred_chord_torch.detach().numpy().reshape(88,))
    next_chord_torch = torch.tensor(next_chord_np, dtype = torch.float).view(1,1,88)
    song_torch = torch.cat((song_torch, next_chord_torch), 1)
    song_np = np.vstack((song_np, next_chord_np))

song_np = song_np.T

chorales.save_to_midi(song_np, 'composition.mid', verbosity=1)
