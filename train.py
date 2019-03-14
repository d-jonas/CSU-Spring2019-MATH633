"""
A module with functions to load saved networks, train the network for a given
number of epochs, then save the resulting network.

An example of implementation would be:

import train
model = train.get_network()
data = chorales.train
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
model = train.train(model,loss_fn,optimizer)
train.save_network(model)

"""

import torch
import LSTM_class
import chorales
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

def get_network(input_size = 88, hidden_size = 25, output_size = 88, num_layers = 1, bidirectional = False, **kwargs):
    """
    Interactively asks the user whether a saved network should be used and
    depending on the answer, will ask for a file name or generate a new network.
    A default filename of saved_net.pt is suggested.
    """
    while True:
        if 'resp' in kwargs:
            resp = kwargs['resp']
        else:
            resp = str(input('Should loading of a saved network be attempted? [y\\n]'))
        #
        if resp == 'y':
            if 'filename' in kwargs:
                name = kwargs['filename']
            else:
                # Define the LSTM network by loading a saved version or creating a new one
                name = str(input('Give the filename or press enter for default (saved_net.pt): '))
            #
            if name == '':
                name = 'saved_net.pt'
            try:
                print('Loading saved network...', end = '')
                network = torch.load(name)
            except:
                print('Unable to load saved network. Creating new network...', end = '')
                network = LSTM_class.LSTM(input_size, hidden_size, output_size, num_layers, bidirectional)
                network.float()
                print('done.')
            break
        elif resp == 'n':
            print('Creating new network...', end = '')
            network = LSTM_class.LSTM(input_size, hidden_size, output_size, num_layers, bidirectional)
            network.float()
            print('done.')
            break
        else:
            print('Sorry, that is not a valid option.')

    return network

def save_network(network):
    """
    Interactively asks the user whether the current network should be saved and,
    depending on the answer, will ask for a file name to which the network
    should be saved. Then, if the filename already exists, the user will be
    prompted to verify that the network should overwrite the current file.
    A default filename of saved_net.pt is suggested.
    """
    resolved = False
    while not resolved:
        resp = str(input('Should saving of the current network be attempted? [y/n] '))
        if resp == 'y':
            # Save the LSTM network
            name = str(input('Give the filename or press enter for default (saved_net.pt): '))
            if name == '':
                name = 'saved_net.pt'
            if Path(name).is_file():
                while True:
                    resp = str(input('Overwrite the file at ' + str(Path(name)) + '? [y/n] '))
                    if resp == 'y':
                        torch.save(network, name)
                        print(f'Network has been saved to {str(Path(name))}.')
                        resolved = True
                        break
                    elif resp == 'n':
                        break
                    else:
                        print('Please answer y or n...')
                        continue
            else:
                torch.save(network, name)
                print(f'Network has been saved to {str(Path(name))}.')
                resolved = True
        elif resp == 'n':
            print('Ok, network will not be saved.')
            resolved = True
        else:
            print('Sorry, that is not a valid option...')

# Prepare data for input into LSTM network
# Pull from chorales, then convert to torch.tensors of correct shape
def get_chorales_tensors(song):
    '''
    Takes a one-hot encoded song and returns an input tensor and target tensor
    Input: numpy array of shape (88, song_len)
    Output: Two torch tensors of shape (song_len - 1, 1, 88)
    '''

    torch_input = torch.tensor(song[:,:-1], dtype=torch.float).view(1, song.shape[1] - 1, -1)
    torch_target = torch.tensor(song[:,1:], dtype=torch.float).view(1, song.shape[1] - 1, -1)

    return torch_input, torch_target

def train(network, loss_fn, optimizer, data, epochs=10, **kwargs):
    # make a plot of the loss?
    vis_loss = kwargs.get('vis_loss', True)

    # Init Loss vector for plotting
    losses = np.empty(epochs)

    # Start timer
    start = time.time()
    print('Training network...')

    # Train network
    for i in range(epochs):
        for song in data:
            torch_tests, torch_tests_targets = get_chorales_tensors2(song)
            network.hidden = network.init_hidden(minibatch_size = song.shape[1]-1)
            out = network.forward(torch_tests)
            loss = loss_fn(out, torch_tests_targets.view(-1,88))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[i] = loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')

    # Plot of loss as a function of epochs
    if vis_loss:
        fig, ax = plt.subplots()
        fig.suptitle('Loss Function: ' + str(loss_fn))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.plot(losses)
        plt.show(block = False)
    #

    return network, losses

def write_song(network, chord):
    '''
    Composes a song from a given input.

    TODO:
    - figure out what input looks like. (1 chord? 2 chords? etc.)
    - figure out threshold to determine how many notes are in an outputted chord.
    - figure out when songs end. (hardcoded T-value?)

    Input:
    Output: np array, or torch tensor, or MIDI list
    '''
    note, note_target = get_chorales_tensors(chorales.train[0][:,:2])
    out = network.forward(note)

    return song

def get_4notes(chord):
    '''
    Input:  chord (LSTM output)
    Output: one-hot encoded chord (with four notes only)
    '''
    temp = np.argpartition(-chord, 4)
    idx = temp[:4] #indices of the first four highest notes
    s = chord.shape[0]
    chord = np.zeros(s)
    chord[idx] = 1

    return chord


loss_library = {
'MSELoss': torch.nn.MSELoss(),
'L1Loss': torch.nn.L1Loss(),
'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
'NLLLoss': torch.nn.NLLLoss()
}
