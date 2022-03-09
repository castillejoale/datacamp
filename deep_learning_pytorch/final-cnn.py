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
import enum
import sys
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import os

class Dataset(enum.Enum):
    mnist = 1
    cifar10 = 2

selected_dataset = Dataset.cifar10

pixels_per_side = 28
number_of_channels = 1

if selected_dataset == Dataset.cifar10:
    pixels_per_side = 32
    number_of_channels = 3

batch_size = 16

if selected_dataset == Dataset.cifar10:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=number_of_channels, out_channels=5, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=5, eps=1e-05, momentum=0.9),
                                      nn.Dropout(p=0.5),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=10, eps=1e-05, momentum=0.9),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=20, eps=1e-05, momentum=0.9),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=0.9),)

        # Declare all the layers for classification
        self.classifier = nn.Sequential(nn.Linear(int(pixels_per_side/4) * int(pixels_per_side/4) * 40, 1024),
                                        nn.ReLU(inplace=True),
                                       	nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048, 10))


    def forward(self, x):

        # Apply the feature extractor in the input
        x = self.features(x)

        # Squeeze the three spatial dimensions in one
        x = x.view(-1, int(pixels_per_side/4) * int(pixels_per_side/4) * 40)

        # Classify the images
        x = self.classifier(x)
        return x


def show_example():
    print("show_example")

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    label = labels[0]
    # Label
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    # print(torch.tensor([label]))
    print(classes[label])

    # Image
    img = images[0]
    # Add dummy variable
    img = img[None, :]
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5     # unnormalize
    print(img.shape)
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(img.permute(1, 2, 0))
    plt.show()



transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485), (0.229))])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), ((0.3081)))])
if selected_dataset == Dataset.cifar10:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), ((0.3081, 0.3081, 0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

if selected_dataset == Dataset.cifar10:
    dataset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10('cifar10', train=False, download=True, transform=transform)

    classes = dataset.classes
    print(classes)

    # Print classes count
    # class_count = {}
    # for _, index in dataset:
    #     label = classes[index]
    #     if label not in class_count:
    #         class_count[label] = 0
    #     class_count[label] += 1
    # print(class_count)

    torch.manual_seed(43)
    val_size = 5000
    train_size = len(dataset) - val_size

    trainset, val_ds = random_split(dataset, [train_size, val_size])
    print(len(train_ds))
    print(len(val_ds))

    # trainset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10('cifar10', train=False, download=True, transform=transform)


    # Prepare training loader and testing loader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

# Compute the shape of the training set and testing set
# trainset_shape = train_loader.dataset.data.shape
# testset_shape = test_loader.dataset.data.shape
# # Print the computed shapes
# print('trainset_shape: ' + str(trainset_shape))
# print('testset_shape: ' + str(testset_shape))
#
# # Compute the size of the minibatch for training set and testing set
# trainset_batchsize = train_loader.batch_size
# testset_batchsize = test_loader.batch_size
# # Print sizes of the minibatch
# print('trainset_batchsize: ' + str(trainset_batchsize))
# print('testset_batchsize: ' + str(testset_batchsize))






def train(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

    epoch_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # print(i)
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

        # print statistics
        epoch_loss += outputs.shape[0] * loss.item() # outputs.shape[0] is the batch size
        running_loss += loss.item()
        print_step = 100
        if i % print_step == print_step - 1:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_step))
            running_loss = 0.0
    # print epoch loss
    print(epoch+1, epoch_loss / len(trainset))


def eval(model):

    model.eval()

    total = 0
    correct = 0

    # Iterate over the data in the test_loader
    for i, data in enumerate(test_loader):
        # print(i)
        # Get the image and label from data
        image, label = data
        # print(image.shape)
        # print(label)
        # sys.exit(0)

        # Make a forward pass in the net with your image
        output = model(image)

        # Argmax the results of the net
        _, predicted = torch.max(output.data, 1)

        total += label.size(0)
        correct += (predicted == label).sum().item()

    print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

def random_prediction(model):
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    label = labels[0]

    img = images[0]
    img = img[None, :]

    predict(model, img, label)


def predict(model, img, label):

    model.eval()

    # Make a forward pass in the net with your image
    output = model(img)

    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)

    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))

model = Net()

# # This brings up testing predictions to 98%
# from urllib.request import urlretrieve
# url = 'https://assets.datacamp.com/production/repositories/4094/datasets/8bf303ea6add66e4ef3298ebf40aea36b6055647/my_net_small.pth'
# urlretrieve(url, 'my_net_small.pth')
# model.load_state_dict(torch.load('my_net_small.pth', map_location='cpu'))
# indices = np.arange(10000)
# np.random.shuffle(indices)
# indices = indices[:100]

# show_example()
random_prediction(model)

num_epochs = 10

for epoch in range(num_epochs):
    train(model)
    eval(model)

random_prediction(model)
