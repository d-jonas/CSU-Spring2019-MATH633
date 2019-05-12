"""
An example script to show how functions in the sessions module are meant to be
used with Pytorch, the custom LSTM class, and the JSB_Chorales data imported
through the chorales module.
"""

import torch
from LSTM import LSTM
import chorales
import sessions

# Number of epochs to train the network
epochs = 10000

# Create a new network
network = LSTM(88, 50, 88, 1) # (input_size, hidden_size, output_size, num_layers)
network.float()

# Create an optimizer and loss function
optimizer = torch.optim.SGD(network.parameters(), lr = 0.02, momentum = 0.5)
loss_fn = torch.nn.MSELoss()

# Train the network on the test data set
network, losses_train, losses_test = sessions.train_with_test(network, loss_fn, optimizer,
                        chorales.train, chorales.test, epochs, minibatch=5)
# Plot the error reduction
trfig, trax = sessions.plot_losses_train_and_test(losses_train, losses_test, epochs)

# Test the network on the test data set and plot results
losses_test = sessions.test(network, loss_fn, chorales.test, minibatch=5)

# Create new song with trained network
new_song = sessions.compose(network, chorales.valid[0][:,0])

# Plot the new song
sessions.plot_song(new_song)

# Save the network
torch.save(network, 'final_net.pt')
