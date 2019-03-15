'''
A script that uses each of the following necessary dependencies to train, validate,
test, and compose songs using the chorales dataset
'''

import torch
import LSTM_class
import chorales
import matplotlib.pyplot as plt
import numpy as np
import time
import train
from train import loss_library

# Init network and parameters
network = train.get_network(
    input_size = 88,
    hidden_size = 25,
    output_size = 88, num_layers = 1,
    bidirectional = False
    )

optimizer = torch.optim.SGD(network.parameters(), lr = 0.5, momentum = 0.5)
data = chorales.train

# Train the network
network, losses = train.train(network, loss_library['MSELoss'], optimizer, data, epochs = 500)

network, song = train.compose_song(network, seed = chorales.train[0][:,:10], song_len = 50)

# chorales.save_to_midi(song, 'composition.mid', verbosity = 1)
