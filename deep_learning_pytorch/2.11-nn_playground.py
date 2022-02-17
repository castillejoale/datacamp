import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from os import chdir
chdir("/Users/alejandrocastillejo/Desktop/datacamp/");

# pixels_per_side = 28
pixels_per_side = 32
# number_of_channels = 1
number_of_channels = 3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(pixels_per_side * pixels_per_side * number_of_channels, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Transform the data to torch tensors and normalize it
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), ((0.3081)))])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), ((0.3081, 0.3081, 0.3081)))])

# Prepare training set and testing set
# trainset = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)
# testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)
trainset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10('cifar10', train=False, download=True, transform=transform)

# Prepare training loader and testing loader
initial_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

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
    # Instantiate the Adam optimizer and Cross-Entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for batch_idx, data_target in enumerate(train_loader):
        data = data_target[0]
        # print(data.shape)
        target = data_target[1]
        data = data.view(-1, pixels_per_side * pixels_per_side * number_of_channels)
        optimizer.zero_grad()

        # Complete a forward pass
        output = model(data)

        # print(data.shape)
        # print(target.shape)
        # print(output.shape)

        # Compute the loss, gradients and change the weights
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def eval(model):

    # Set the model in eval mode
    model.eval()

    total = 0
    correct = 0

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data

        # Put each image into a vector
        inputs = inputs.view(-1, pixels_per_side*pixels_per_side*number_of_channels)

        # Do the forward pass and get the predictions
        outputs = model(inputs)
        _, outputs = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (outputs == labels).sum().item()

    print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

model = Net()

train(model)
eval(model)
