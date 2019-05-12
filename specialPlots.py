"""
Functions which create plots used in the writing of my paper with Ian for
Math 633 at CSU during Spring 2019.
"""

import torch
from LSTM import LSTM
import time
import numpy as np
import matplotlib.pyplot as plt

def batch_song(song, minibatch):
    """
    Reshapes song into a torch tensor of appropriate dimensions given the
    minibatch size.
    """
    batch = np.zeros((song.size//88//minibatch-1, minibatch, 88))
    targets = np.zeros((song.size//88//minibatch-1, minibatch, 88))
    for i in range(song.size//88//minibatch-1):
        batch[i,:,:] = song[:,i:i+minibatch].T
        targets[i] = song[:,i+1:i+minibatch+1].T
    batch = torch.tensor(batch,dtype=torch.float)
    targets = torch.tensor(targets,dtype=torch.float)
    return batch, targets

def vary_layers(song):
    """
    Plots the error reduction on one song for multiple hidden layer depths
    """
    epochs = 10000
    minibatch = 4

    # 1 hidden layer
    network = LSTM(88, 50, 88, 1) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses_hid1 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()

    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_hid1[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # 2 hidden layers
    network = LSTM(88, 50, 88, 2) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses_hid2 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()
    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_hid2[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # 4 hidden layers
    network = LSTM(88, 50, 88, 4) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses_hid4 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()
    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_hid4[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # 8 hidden layers
    network = LSTM(88, 50, 88, 8) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses_hid8 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()
    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_hid8[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # Produce plot
    fig, ax = plt.subplots()
    ax.set_title('Error Reduction for Different Hidden Layer Depths (HLD)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    t = list(range(1,epochs+1))
    ax.plot(t, losses_hid1, t, losses_hid2, t, losses_hid4, t, losses_hid8)
    ax.legend(['HLD = 1','HLD = 2','HLD = 4','HLD = 8'])
    plt.show(block = False)


def vary_size(song):
    """
    Produce a plot of error reduction for various hidden layer sizes.
    """
    epochs = 10000
    minibatch = 4

    # Hidden layer size 25
    network = LSTM(88, 25, 88, 1) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses25 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()
    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses25[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # Hidden layer size 50
    network = LSTM(88, 50, 88, 1) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses50 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()
    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses50[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # Hidden layer size 100
    network = LSTM(88, 100, 88, 1) # (input_size, hidden_size, output_size, num_layers)
    network.float()

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
    loss_fn = torch.nn.MSELoss()

    losses100 = np.zeros(epochs)

    batch, targets = batch_song(song, minibatch)

    start = time.time()
    for i in range(epochs):
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses100[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')


    # Produce plot
    fig, ax = plt.subplots()
    ax.set_title('Error Reduction for Different Hidden Layer Sizes (HLS)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    t = list(range(1,epochs+1))
    ax.plot(t, losses25, t, losses50, t, losses100)
    ax.legend(['HLS = 25','HLS = 50','HLS = 100'])
    plt.show(block = False)
