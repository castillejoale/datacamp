import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Patch torch.load
def path_torch_load(path):
    # This function will patch torch.load to just return the path of the original model.
    if type(path)==str:
        return path
    else:
        raise TypeError('Please input a path to an existing model.')

torch.load = path_torch_load

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # instantiate all 3 linear layers
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc = nn.Linear(7 * 7 * 512, 10)
        # Set dummy parameters
        self.train_mode = False
        self.is_trained = False
        self.previous_state_loaded = False

    def load_state_dict(self, path):
        if type(path)==str:
            self.previous_state_loaded = True
        else:
            raise TypeError('Please input a path to an existing model.')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 512)
        return self.fc(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        # Was fc entered correctly?
        if type(self.fc) == type(nn.Linear(7 * 7 * 512, 26)):
            # Were the number of out channel changes?
            if self.fc.out_features == 26:
                # Was the model set to train mode?
                if self.train_mode:
                    # Was the model actually trained?
                    if self.is_trained:
                        # Is this the previously trained model?
                        if self.previous_state_loaded:
                            return 0.84
                            # This is the naieve model
                        else:
                            return 0.57
                    else:
                        raise ValueError('Did you remember to train your model?')
                else:
                    raise ValueError('Did you remember to set your model to train mode?')
            else:
                raise ValueError('There should be 26 out channels for the 26 letters of the alphabet.')
        else:
            raise ValueError('Did you remember to defined model.fc?')

def train_net(model, optimizer, criterion):
    # Check that model is a Net
    if type(model) == type(Net()):
        # Check that optimizer is an Adam
        if type(optimizer) == type(optim.Adam(model.parameters(), lr=3e-4)):
            # Check that criterion is CrossEntropyLoss
            if type(criterion) == type(nn.CrossEntropyLoss()):
                model.is_trained = True
            else:
                raise TypeError('criterion should be of type CrossEntropyLoss.')
        else:
            raise TypeError('optimizer should be of type  Adam Optimizer.')
    else:
        raise TypeError('model should be of type Net().')



model = Net()
# instantiate the Adam optimizer and Cross-Entropy loss function
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# Create a model using
model = Net()

# Load the parameters from the old model
model.load_state_dict(torch.load('my_net.pth'))

# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))
