import torch
import LSTM_class
import chorales
import matplotlib.pyplot as plt
import numpy as np
import time

# Prepare data for input into LSTM network
# Pull from chorales, then convert to torch.tensors of correct shape
def get_chorales_tensors(song):
    '''
    Takes a one-hot encoded song and returns an input tensor and target tensor
    Input: numpy array of shape (88, song_len - 1)
    Output: Two torch tensors of shape (song_len - 1, 1, 88)
    '''
    torch_input = torch.tensor(song[:,:-1],dtype=torch.float).view(song.shape[1] - 1, 1, -1)
    torch_target = torch.tensor(song[:,1:],dtype=torch.float).view(song.shape[1] - 1, 1, -1)

    return torch_input, torch_target

# Define the LSTM network
network = LSTM_class.LSTM(input_size = 88, output_size = 88)
network.float()

# Define the loss function and optimization function
loss_library = {
'MSELoss': torch.nn.MSELoss(),
'L1Loss': torch.nn.L1Loss(),
'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
'NLLLoss': torch.nn.NLLLoss()
}

loss_fn = loss_library['MSELoss']
optimizer = torch.optim.SGD(network.parameters(), lr=0.05, momentum=0.5)

# Number of epochs
epochs = 100

# Init Loss vector for plotting
losses = np.empty(epochs)

# start timer
start = time.time()

# THIS LOOP ITERATES THROUGH ONE SONG ##
torch_tests, torch_tests_targets = get_chorales_tensors(chorales.train[0])
for i in range(epochs):
    network.hidden = network.init_hidden()
    out = network.forward(torch_tests)
    loss = loss_fn(out, torch_tests_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses[i] = loss.item()

    # occasionally print the loss
    if i%5 == 0:
        print("Epoch: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')

# # THIS LOOP ITERATES THROUGH ENTIRE TRAINING DATASET ##
# for i in range(epochs):
#     for song in chorales.train:
#         torch_tests, torch_tests_targets = get_chorales_tensors(song)
#         network.hidden = network.init_hidden()
#         out = network.forward(torch_tests)
#         loss = loss_fn(out, torch_tests_targets.view(128,88))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     losses[i] = loss.item()
#
#     # occasionally print the loss
#     if i%5 == 0:
#         print("Round: " + str(i) + "/" + str(epochs) + "; Error: " + str(loss.item()), end='\r')
# #

end = time.time()
print('Total Duration: ' + str((end - start)/60) + ' minutes')

# quick plot of loss as a function of epoch
fig, ax = plt.subplots()
fig.suptitle('Loss Function: ' + str(loss_fn))
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.plot(losses)
plt.show(block = False)
