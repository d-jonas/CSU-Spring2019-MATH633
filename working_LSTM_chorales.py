import torch
import LSTM_class
import chorales
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for input into LSTM network
# Pull from chorales, then convert to torch.tensors of correct shape
def get_chorales_tensors(song):
    '''
    Takes a one-hot encoded song and returns an input tensor and target tensor
    Input: numpy array of shape (88, song_len - 1)
    Output: Two torch tensors of shape (song_len - 1, 1, 88)
    '''
    torch_input = torch.tensor(song[:,:-1]).view(song.shape[1] - 1, 1, -1).float()
    torch_target = torch.tensor(song[:,1:]).view(song.shape[1] - 1, 1, -1).float()

    return torch_input, torch_target

torch_tests, torch_tests_targets = get_chorales_tensors(chorales.train[0])


# Define the LSTM network
network = LSTM_class.LSTM(input_size = 88, output_size = 88)
network.float()

# Define the loss function and optimization function
# Potential Loss Fucntions: L1Loss, MSELoss, CrossEntropyLoss, NLLLoss
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.05, momentum=0.5)

# Number of epochs
epochs = 500

# Init Loss vector for plotting
losses = np.empty(epochs)

# Start training the network
for i in range(epochs):
    network.hidden = network.init_hidden()
    out = network.forward(torch_tests, torch_tests.shape[0])
    loss = loss_fn(out, torch_tests_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses[i] = loss.item()

    # Occasionally print the loss
    if i%5 == 0:
        print("Round: ", i, "; MSE: ", loss.item(), end='\r')
