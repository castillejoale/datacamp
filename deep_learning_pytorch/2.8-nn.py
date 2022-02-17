# Import torch and torch.nn
import torch
import torch.nn as nn

# Define the class Net
class Net(nn.Module):
    def __init__(self):
    	# Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
