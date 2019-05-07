"""
A collection of functions which implement training, testing, and composing
phases of the network use, along with some functions to create plots of the
loss reduction or new songs.
"""

import torch
import time
import chorales
import numpy as np
import matplotlib.pyplot as plt

def train(network, loss_fn, optimizer, collection, epochs=1000, minibatch=5):
    """
    Performs a training session on the given collection for a chosen number of
    epochs, prints some loss information as the session progresses, and
    produces a numpy array for plotting of the loss progress at the end of the
    session. Returns a trained network.
    """
    losses = np.zeros(epochs)
    start = time.time()
    for i in range(epochs):
        for song in collection:
            batch = np.zeros((song.size//88//minibatch-1, minibatch, 88))
            targets = np.zeros((song.size//88//minibatch-1, minibatch, 88))
            network.hidden = network.init_hidden(minibatch_size = minibatch)
            for j in range(song.size//88//minibatch-1):
                batch[j,:,:] = song[:,j:j+minibatch].T
                targets[j] = song[:,j+1:j+minibatch+1].T
            batch = torch.tensor(batch,dtype=torch.float)
            targets = torch.tensor(targets,dtype=torch.float)
            out = network.forward(batch)
            loss = loss_fn(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[i] += loss.item()



        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    losses = losses/len(chorales.train)

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')
    return network, losses

def train_with_test(network, loss_fn, optimizer, train_collection, test_collection,
          epochs=1000, minibatch=5):
    """
    Performs a training session on the given train collection for a chosen number of
    epochs, prints some loss information as the session progresses, and
    produces a numpy array for plotting of the loss progress at the end of the
    session. Returns a trained network. Once every epoch, the network is passed
    the test collection and the result is stored in a numpy array for plotting
    later.
    """
    losses_train = np.zeros(epochs)
    losses_test = np.zeros(epochs)
    start = time.time()
    for i in range(epochs):
        for song in train_collection:
            batch = np.zeros((song.size//88//minibatch-1, minibatch, 88))
            targets = np.zeros((song.size//88//minibatch-1, minibatch, 88))
            network.hidden = network.init_hidden(minibatch_size = minibatch)
            for j in range(song.size//88//minibatch-1):
                batch[j,:,:] = song[:,j:j+minibatch].T
                targets[j] = song[:,j+1:j+minibatch+1].T
            batch = torch.tensor(batch,dtype=torch.float)
            targets = torch.tensor(targets,dtype=torch.float)
            out = network.forward(batch)
            loss = loss_fn(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_train[i] += loss.item()

        for song in test_collection:
            batch = np.zeros((song.size//88//minibatch-1, minibatch, 88))
            targets = np.zeros((song.size//88//minibatch-1, minibatch, 88))
            network.hidden = network.init_hidden(minibatch_size = minibatch)
            for j in range(song.size//88//minibatch-1):
                batch[j,:,:] = song[:,j:j+minibatch].T
                targets[j] = song[:,j+1:j+minibatch+1].T
            batch = torch.tensor(batch,dtype=torch.float)
            targets = torch.tensor(targets,dtype=torch.float)
            out = network.forward(batch)
            loss = loss_fn(out, targets)
            losses_test[i] += loss.item()

        # Occasionally print the loss
        if i%5 == 0:
            print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

    losses_train = losses_train/len(chorales.train)
    losses_test = losses_test/len(chorales.test)

    end = time.time()
    print('\nTraining successful!')
    print('Total Duration: ' + str((end - start)/60) + ' minutes')
    return network, losses_train, losses_test

def test(network, loss_fn, collection, minibatch=5):
    """
    Performs a test session on the given collection, computing the difference
    between the predictions and the targets and plotting the losses for
    all songs individually.
    """
    losses = np.zeros(len(collection))
    index = 0
    for song in collection:
        batch = np.zeros((song.size//88//minibatch-1, minibatch, 88))
        targets = np.zeros((song.size//88//minibatch-1, minibatch, 88))
        network.hidden = network.init_hidden(minibatch_size = minibatch)
        for j in range(song.size//88//minibatch-1):
            batch[j,:,:] = song[:,j:j+minibatch].T
            targets[j] = song[:,j+1:j+minibatch+1].T
        batch = torch.tensor(batch,dtype=torch.float)
        targets = torch.tensor(targets,dtype=torch.float)
        out = network.forward(batch)
        loss = loss_fn(out, targets)
        losses[index] = loss.item()
        index += 1
    fig, ax = plt.subplots()
    ax.set_title('Error Applied to Test Data')
    ax.set_xlabel('Song')
    ax.set_ylabel('Error')
    t = list(range(1,len(collection)+1))
    ax.plot(t, losses)
    plt.show(block = False)
    return losses

def get_4_notes(chord):
    '''
    Input:  chord (LSTM output). Must first be converted to a 1-D np array
    Output: one-hot encoded chord (with four notes only)
    Used to map LSTM predictions to chords in the compose function.
    '''
    temp = np.argpartition(-chord, 4)
    idx = temp[:4] #indices of the first four highest notes
    s = chord.shape[0]
    chord = np.zeros(s)
    chord[idx] = 1

    return chord

def compose(network, first_chord, song_len=100):
    """
    Composes a new song based on the given seed and the network's predictions.
    Length defaults to 100, but can be given.
    """
    # Init songs in both np.array and torch.tensor format
    song_np = [first_chord]
    song_torch = torch.tensor(first_chord).view(1, 1, 88).float()
    network.hidden = network.init_hidden(1)

    for i in range(song_len):
        pred_chord_torch = network.forward(song_torch)[:,-1,:] # predict next note w/ probabilities
        next_chord_np = get_4_notes(pred_chord_torch[-1].detach().numpy().reshape(88,)) #->np.array of probabilities -> one hot encoded vector
        next_chord_torch = torch.tensor(next_chord_np, dtype = torch.float).view(1,1,88) # np.array -> torch.tensor
        song_torch = torch.cat((song_torch, next_chord_torch), 0) # append to torch tensor song
        song_np = np.append(song_np, next_chord_np.reshape(1,88), axis = 0) # append to np.array song

    chorales.save_to_midi(song_np.T, 'composition.mid', verbosity = 1)
    return song_np


def plot_losses(losses, epochs):
    """
    Produces a plot of the loss function values stored in losses.
    """
    fig, ax = plt.subplots()
    ax.set_title('Error Reduction')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    t = list(range(1,epochs+1))
    ax.plot(t, losses)
    plt.show(block = False)

def plot_losses_train_and_test(losses_train, losses_test, epochs):
    """
    Produces a plot of the loss function values stored in losses_train and
    losses_test.
    """
    fig, ax = plt.subplots()
    ax.set_title('Error Reduction')
    ax.set_xlabel('Error on Training Dataset')
    ax.set_ylabel('Error on Test Dataset')
    t = list(range(1,epochs+1))
    ax.plot(t, losses_train, color='b', label='Train')
    ax.plot(t, losses_test, color='r', label='Test')
    ax.legend()
    plt.show(block = False)

def plot_song(song):
    """
    Produces a plot of the notes in each chord of the song, with chord number
    on the x-axis and note name on the y-axis.
    """
    fig,ax = plt.subplots(1,1, figsize=(12,5))

    ax.imshow(song.T, cmap=plt.cm.Greys )

    # Prettify the plot a little. Assume everything is in 4/4 time.
    ax.set_xticks( [8*j for j in range(song.T.shape[1]//8 + 1)] )

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

    plt.yticks(range(shift,shift+88,skip),tone_labels)

    ax.set_xlabel('Beat number')
    ax.set_ylabel('Tone')
    ax.xaxis.grid()
    ax.invert_yaxis()
    plt.title('A Sequence of Chords')

    plt.show(block = False)
