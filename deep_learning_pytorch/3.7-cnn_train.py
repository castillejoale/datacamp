import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from os import chdir
chdir("/Users/alejandrocastillejo/Desktop/datacamp/");
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)


    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)








net = Net()
from urllib.request import urlretrieve
url = 'https://assets.datacamp.com/production/repositories/4094/datasets/8bf303ea6add66e4ef3298ebf40aea36b6055647/my_net_small.pth'
urlretrieve(url, 'my_net_small.pth')
net.load_state_dict(torch.load('my_net_small.pth', map_location='cpu'))
indices = np.arange(10000)
np.random.shuffle(indices)
indices = indices[:100]


# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485), (0.229))])
trainset = datasets.MNIST('mnist', train=True, download=True,transform=transform)
# trainset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset,batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices), num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)






for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)

    # Compute the loss function
    loss = criterion(outputs, labels)

    # Compute the gradients
    loss.backward()

    # Update the weights
    optimizer.step()
