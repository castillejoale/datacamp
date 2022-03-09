import torch.nn as nn
# Import the module
import torchvision

# Download resnet18
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the layers bar the last one
for param in model.parameters():
    param.requires_grad = False

# Change the number of output units
model.fc = nn.Linear(512, 7)
