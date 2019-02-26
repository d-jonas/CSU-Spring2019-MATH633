import torch
import chorales

"""
A script to help us sort out what is supposed to be done by the forward function
in LSTM_class.py. Add comments and generality as you learn things.
"""

# Defining parameters for the data and LSTM block
input_size = 88
hidden_size = 88
num_layers = 1
output_size = 88

# Pick out first chorale from dictionary imported with chorales
song = torch.tensor(chorales.train[0],dtype=torch.uint8) # An 88x129 tensor


# An instance of pytorch's LSTM class
model = torch.nn.LSTM(input_size = input_size,
                      hidden_size = hidden_size,
                      num_layers = num_layers,
                      bias = True,
                      batch_first = False,
                      dropout = 0,
                      bidirectional = False)

# First dimension is 2 if bidirectional = True
hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))

# Output layer to take LSTM block output and convert to a prediction
# 2*hidden_size if bidirectional
out_layer = torch.nn.Linear(hidden_size, output_size)

# Initialize a random input
input = torch.randn(1,1,input_size)

# Initialize a tensor of the proper size containing song
input = torch.zeros(len(song[0]),1,input_size)
input[:,0,:] = song.permute(1,0)

# Get LSTM block output and updated hidden weights
out,hidden = model(input,hidden)

print(out.size()) # (1,1,2*hidden_size)
print(hidden[0].size()) # (2,1,hidden_size)
print(hidden[1].size()) # (2,1,hidden_size)

pred = out_layer(out)

print(pred.size()) # (1,1,88)
print(pred) # Output isn't integers... don't know what to do here
