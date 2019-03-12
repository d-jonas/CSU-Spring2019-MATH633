'''
Purpose: study the amount of memory our LSTM class has.
First question is if the LSTMs have the Markov property (memorylessness)?

This would be the case if, for any two inputs "input1" and "input2",

_ = network1.forward(input1)
output1 = network1.forward(input2)

output2 = network2.forward(input2)

||output1 - output2|| -> 0

This is not the case (not done here, but I've checked this).
The next question is the *amount* of memory an LSTM has. For example,
if one LSTM gets fed a sequence of inputs, and the other one misses
the first few terms, do they eventually align? Or is it in
finite iterations (roughly, a short-term recurrence)? For instance,
if it is a one-term recurrence, then we would see

_ = network1.forward(input1)
_ = network1.forward(input2)
output1 = network1.forward(input3)

_ = network2.forward(input2)
output2 = network2.forward(input3)

||output1 - output2|| -> 0

-Nuch
March 7, 2019
'''

import torch
import chorales
import numpy as np
import train

import copy

from matplotlib import pyplot

loss_library = {
'MSELoss': torch.nn.MSELoss(),
'L1Loss': torch.nn.L1Loss(),
'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
'NLLLoss': torch.nn.NLLLoss()
}

def train_with_layers(num_layers):

    # Init network and parameters
    network = train.get_network(
        input_size = 88,
        hidden_size = 25,
        output_size = 88,
        num_layers = num_layers,
        bidirectional = False,
        resp='n'
        )

    optimizer = torch.optim.SGD(network.parameters(), lr = 0.5, momentum = 0.5)
    data = chorales.train


    # Train the network
    network, losses = train.train(network, loss_library['MSELoss'], optimizer, data, epochs = 1, vis_loss=False)

    return network
#

# Train two networks, with one and two layers.
# We're trying to see if the number of layers indicates
# the amount of memory the network has.
net_l1 = train_with_layers(1)
net_l2 = train_with_layers(2)

# Now we want to study each network. With a sequence of
# inputs, how much memory (or lack of it) does each have?
# Will two networks give the same output if exposed to
# one prior entry? two prior entries?
which = net_l2

max_inputs = 8
inputs = [torch.tensor(np.random.rand(128,1,88)).float() for _ in range(max_inputs)]
networks = [copy.copy(which) for _ in range(max_inputs)]

net_orig = copy.copy(which)

hidden_states = np.zeros((max_inputs,max_inputs))
hidden_states[:,0] = [float( network.hidden[1][0,0,0].detach().numpy() ) for network in networks]
for j,input in enumerate(inputs):
    # Give this input only to the first j networks.
    for i in range(j):
        _ = networks[i].forward(input)
    # But record a single entry in the .hidden variables.
    # This will be an indicator if the collection of networks update their
    # hidden variables in a way which has a memory of more than the most recent
    # input.
    for i in range(max_inputs):
        hidden_states[i,j] = float( networks[i].hidden[1][0,0,0].detach().numpy() )
#

fig,ax = pyplot.subplots(1,1)
ax.imshow(hidden_states, cmap=pyplot.cm.rainbow)

ax.set_xlabel('Iteration number', fontsize=14)
ax.set_ylabel('Network', fontsize=14)

fig.suptitle('LSTM hidden state as function of its prior inputs', fontsize=16)

fig.tight_layout()
fig.subplots_adjust(top=0.9)
fig.show()
