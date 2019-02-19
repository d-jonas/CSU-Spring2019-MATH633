import torch

"""
A script to help us sort out what is supposed to be done by the forward function
in LSTM_class.py. Add comments and generality as you learn things.
"""

# Defining parameters for the data and LSTM block
input_size = 88
hidden_size = 100
num_layers = 1
output_size = 88

# An instance of pytorch's LSTM class
model = torch.nn.LSTM(input_size = input_size,
                      hidden_size = hidden_size,
                      num_layers = num_layers,
                      bias = True,
                      batch_first = False,
                      dropout = 0,
                      bidirectional = True)

# Why does the first dimension have to be 2???
hidden = (torch.zeros(2,1,hidden_size),torch.zeros(2,1,hidden_size))

# Output layer to take LSTM block output and convert to a prediction
out_layer = torch.nn.Linear(2*hidden_size, output_size)

# Initialize a random input
input = torch.randn(1,1,input_size)

# Get LSTM block output and updated hidden weights
out,hidden = model(input,hidden)

print(out.size()) # (1,1,2*hidden_size)
print(hidden[0].size()) # (2,1,hidden_size)
print(hidden[1].size()) # (2,1,hidden_size)

pred = out_layer(out)

print(pred.size()) # (1,1,88)
print(pred)
