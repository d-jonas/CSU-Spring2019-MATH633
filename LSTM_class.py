import torch

"""Initializes an LSTM neural network with default parameters defined by the
   constants predefined in this script.""" 

class LSTM(torch.nn.Module):
    def __init__(self, num_samples, input_size=87, num_nodes=20, 
                 output_size=87, num_layers=1, bidirectional=False):
        super().__init__() # Run Module class constructor
        self.input_size = input_size
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Define LSTM layers using pytorch built-in
        self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                                  hidden_size = self.num_nodes, 
                                  num_layers = self.num_layers,
                                  bias = True,
                                  batch_first = False,
                                  dropout = 0,
                                  bidirectional = self.bidirectional)
        
        # Define the final transformation from the hidden layers to output
        self.linear = torch.nn.Linear(self.num_nodes, output_size)
    
    def forward(self, input):
        """Runs input data once through the network."""
        # View reshapes the input into the expected size
        # Probably should be done outside of the class            
        input = input.view(self.input_size, self.num_samples, 
                           self.num_nodes)
        
        # Send input through hidden LSTM layers
        out, self.hidden = self.lstm(input, self.hidden)
                                     
        # Convert output of LSTM layers into final output prediction
        pred = self.linear(out.view(self.input_size, self.num_samples,
                                    self.num_nodes))
        
        # Return prediction of size output_size
        return pred