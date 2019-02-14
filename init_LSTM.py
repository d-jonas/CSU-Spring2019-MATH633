import torch

"""Initializes an LSTM neural network with default parameters defined by the
   constants predefined in this script.""" 

# Define default parameters for LSTM model based on expected data.
INPUT_SIZE = 15       # # of features in the input vector
NUM_NODES = 12        # Number of learnable characteristics
NUM_LAYERS = 1        # Number of hidden layers
OUTPUT_SIZE = 1       # Length of output vector
NUM_SAMPLES = 200     # Length of input sequence (# of samples)
LEARN_RATE = 0.1      # Learning rate

def main(input_size = INPUT_SIZE, num_nodes = NUM_NODES, 
         num_layers = NUM_LAYERS, output_size = OUTPUT_SIZE, 
         num_samples = NUM_SAMPLES, learn_rate = LEARN_RATE):
    """Initializes an LSTM neural network which is returned as output."""
    
    class LSTM(torch.nn.Module):
        def __init__(self, input_size, num_nodes, num_samples, output_size,
                     num_layers):
            super().__init__() # Run Module class constructor
            self.input_size = input_size
            self.num_nodes = num_nodes
            self.num_samples = num_samples
            self.num_layers = num_layers
            self.bidirectional = True
            
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
            
        def default_initial(self):
            # Generates default inital weights for the hidden layers and cells
            initial_hidden = torch.zeros(self.num_layers, self.num_samples,
                                         self.num_nodes)
            initial_cell = torch.zeros(self.num_layers, self.num_samples,
                                       self.num_nodes)
            return(initial_hidden, initial_cell)
    
    # Create an instance of the LSTM network class
    network = LSTM(input_size, num_nodes, num_samples, output_size,
                     num_layers)
    
    # Create an instance of the loss function
    loss_func = torch.nn.MSELoss()
    
    # Creat an instance of the optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr = learn_rate)
    
    return(network, loss_func, optimizer)
            