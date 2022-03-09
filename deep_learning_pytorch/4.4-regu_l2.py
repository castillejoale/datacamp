import torch
import torch.nn as nn
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):
        # Apply conv followd by relu, then in next line pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Apply conv followd by relu, then in next line pool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)
        # Apply the fully connected layer and return the result
        return self.fc(x)



# Instantiate the network
model = Net()

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
