# import torch.nn as nn\\nimport torch.optim as optim\\nimport torch.utils.data\\nimport torchvision\\nimport torchvision.transforms as transforms\\nimport torch.nn.functional as F\\n\\nfrom os import chdir\\nchdir(&#39;/usr/local/share/datasets/&#39;)\\n\\nclass Net(nn.Module):\\n    def __init__(self):\\n        super(Net, self).__init__()\\n        self.fc1 = nn.Linear(28 * 28, 200)\\n        self.fc2 = nn.Linear(200, 10)\\n\\n    def forward(self, x):\\n        x = F.relu(self.fc1(x))\\n        return self.fc2(x)\\n\\n\\ntransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, 0.1307, 0.1307), ((0.3081, 0.3081, 0.3081)))])\\n\\n# Prepare training set and testing set\\ntrainset = torchvision.datasets.MNIST(&#39;mnist&#39;, train=True, download=True, transform=transform)\\ntestset = torchvision.datasets.MNIST(&#39;mnist&#39;, train=False, download=True, transform=transform)\\n\\n# Prepare training loader and testing loader\\ninitial_train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)\\ntestloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)\\n\\n# This shrinks the training dataset to something DataCamp can manage.\\ntrain_loader = []\\nfor batch_idx, (data, target) in enumerate(initial_train_loader):\\n  if batch_idx &lt; 50:\\n    train_loader.append((data, target))\\n  else:\\n    break


import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from os import chdir
chdir("/Users/alejandrocastillejo/Desktop/datacamp/");

pixels_per_side = 28
number_of_channels = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(pixels_per_side * pixels_per_side * number_of_channels, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Transform the data to torch tensors and normalize it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

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


# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()

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
