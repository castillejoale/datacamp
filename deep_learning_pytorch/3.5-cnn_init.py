import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from os import chdir
# chdir("/Users/alejandrocastillejo/Desktop/datacamp/");

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)

        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU(inplace=True)

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Instantiate a fully connected layer
        self.fc = nn.Linear(490, 10)
