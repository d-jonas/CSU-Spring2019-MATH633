import torch
import LSTM_class
import chorales
import matplotlib.pyplot as plt
import numpy as np
import time

from matplotlib import pyplot

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
optimizer = torch.optim.SGD(network.parameters(), lr=1, momentum=0.5)

# Number of epochs
epochs = 1001

# Init Loss vector for plotting
# losses = np.empty(epochs)
losses = []
preds = []

# start timer
start = time.time()

# THIS LOOP ITERATES THROUGH ONE SONG ##
torch_tests, torch_tests_targets = get_chorales_tensors(chorales.train[0])

def iter_epoch():
    network.hidden = network.init_hidden()
    out = network.forward(torch_tests, torch_tests.shape[0])
    loss = loss_fn(out, torch_tests_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append( loss.item() )
#

for i in range(epochs):
    iter_epoch()
    if i%(epochs//5)==0:
        p = network.forward(torch_tests, torch_tests.shape[0])
        p = p.detach().numpy().reshape(p.shape[0], p.shape[2])
        preds.append( p )
        print(i, ':', losses[-1])
#

fig,ax = pyplot.subplots(1, len(preds), sharex=True, sharey=True, figsize=(15,3))

# examine thresholding
p_last = preds[-1]
for i in range(1,len(preds)):
    thresh = p_last.max() - 2.**(-i)
    ax[i].imshow(p_last.T > thresh, vmin=0.1, vmax=0.8, cmap=pyplot.cm.Greys)
    ax[i].set_title(r'$value > %.1e$'%(thresh,))
#

ground_truth = torch_tests.detach().numpy().reshape((torch_tests.shape[0],torch_tests.shape[2]))
ax[0].imshow(ground_truth.T, vmin=0.1, vmax=0.8, cmap=pyplot.cm.Greys)
ax[0].set_title('Ground truth')

fig.suptitle('Epoch %i prediction with various thresholds'%epochs)
fig.tight_layout()
fig.show()

pyplot.ion()
