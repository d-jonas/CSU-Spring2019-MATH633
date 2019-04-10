import torch

"""Initializes an LSTM neural network class with convenient default
   parameters for chorales."""

class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=25, output_size=1, num_layers=1,
                 bidirectional=False):
        super().__init__() # Run Module class constructor
        self.input_size = input_size # Number of expected features in the input
        self.hidden_size = hidden_size # Number of features in hidden state
        self.output_size = output_size # Number of expected features in the output
        self.num_layers = num_layers # number of LSTM blocks to 'stack'
        self.bidirectional = bidirectional

        # Define LSTM layers using pytorch built-in
        self.lstm = torch.nn.LSTM(input_size = self.input_size,
                                  hidden_size = self.hidden_size,
                                  num_layers = self.num_layers,
                                  bias = True,
                                  batch_first = False,
                                  dropout = 0,
                                  bidirectional = self.bidirectional)

        # Define the final transformation from the hidden layers to output
        self.linear = torch.nn.Linear(self.hidden_size, output_size)

        # Initialize hidden weights
        self.hidden = self.init_hidden()

    def init_hidden(self, minibatch_size = 1):
        # The axes are (num_layers, minibatch_size, hidden_size)
        return (torch.zeros(self.num_layers, minibatch_size, self.hidden_size),
                torch.zeros(self.num_layers, minibatch_size, self.hidden_size))

    def forward(self, input):
        """
        Runs input data once through the network.
        """

        # Send input through hidden LSTM layers
        out, self.hidden = self.lstm(input, self.hidden)

        # Convert output of LSTM layers into final output prediction
        pred = self.linear(out)

        # Return prediction of size output_size
        return pred
