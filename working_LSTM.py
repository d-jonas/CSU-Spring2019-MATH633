import torch
import LSTM_class
import td
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for input into LSTM network
tests = []
tests_targets = []
for i in range(len(td.td)):
    tests.append(td.td[i][0])
    tests_targets.append(td.td[i][1])

torch_tests = torch.tensor(tests).view(-1,1,1)
torch_tests_targets = torch.tensor(tests_targets).view(-1,1,1)

# Define the LSTM network
network = LSTM_class.LSTM()
network.float()

# Define the loss function and optimization function
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.05, momentum=0.5)

# Start training the network
for i in range(1000):
    network.hidden = network.init_hidden()
    out = network.forward(torch_tests,len(tests))
    loss = loss_fn(out, torch_tests_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Occasionally print the loss
    if i%5 == 0:
        print("Round: ", i, "; MSE: ", loss.item(), end='\r')

# Try making predictions
x = np.arange(0,6.28,0.1)
y = []
pred = network.forward((torch.tensor(x,dtype=torch.float).view(-1,1,1)),len(x))
y = [i for i in pred.view(-1,1)]
plt.plot(x,y)
x_true = [point[0] for point in td.td]
y_true = [point[1] for point in td.td]
plt.plot(x_true,y_true)
plt.title(f'working_LSTM.py prediction (MSE: {loss.item()})')
plt.xlabel('inputs')
plt.ylabel('outputs')
plt.show()
