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

# pixels_per_side = 28
pixels_per_side = 32
# number_of_channels = 1
number_of_channels = 3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=number_of_channels, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Instantiate a fully connected layer
        self.fc = nn.Linear(int(pixels_per_side/4) * int(pixels_per_side/4)  * 10, 10)


    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, int(pixels_per_side/4)  * int(pixels_per_side/4) * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)


# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485), (0.229))])
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), ((0.3081)))])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), ((0.3081, 0.3081, 0.3081)))])


# Prepare training set and testing set
# trainset = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)
# testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10('cifar10', train=False, download=True, transform=transform)

# Prepare training loader and testing loader
initial_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# Compute the shape of the training set and testing set
trainset_shape = initial_train_loader.dataset.data.shape
testset_shape = test_loader.dataset.data.shape
# Print the computed shapes
# print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = initial_train_loader.batch_size
testset_batchsize = test_loader.batch_size
# Print sizes of the minibatch
# print(trainset_batchsize, testset_batchsize)

# This shrinks the training dataset to something DataCamp can manage.
train_loader = []
for batch_idx, (data, target) in enumerate(initial_train_loader):
    if batch_idx < 50:
        train_loader.append((data, target))
    else:
        break


def train(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Compute the forward pass
        outputs = model(inputs)

        # Compute the loss function
        loss = criterion(outputs, labels)

        # Compute the gradients
        loss.backward()

        # Update the weights
        optimizer.step()


def eval(model):

    model.eval()

    total = 0
    correct = 0

    # Iterate over the data in the test_loader
    for i, data in enumerate(test_loader):

        # Get the image and label from data
        image, label = data

        # Make a forward pass in the net with your image
        output = model(image)

        # Argmax the results of the net
        _, predicted = torch.max(output.data, 1)

        # if predicted == label:
        #     print("Yipes, your net made the right prediction " + str(predicted))
        # else:
        #     print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))

        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))


model = Net()

# # This brings up testing predictions to 98%
# from urllib.request import urlretrieve
# url = 'https://assets.datacamp.com/production/repositories/4094/datasets/8bf303ea6add66e4ef3298ebf40aea36b6055647/my_net_small.pth'
# urlretrieve(url, 'my_net_small.pth')
# model.load_state_dict(torch.load('my_net_small.pth', map_location='cpu'))
# indices = np.arange(10000)
# np.random.shuffle(indices)
# indices = indices[:100]

train(model)
eval(model)
