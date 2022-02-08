# Import torch
import torch
import torch.nn as nn
torch.manual_seed(314)

input_layer = torch.randn(1, 4)
weight_1 = torch.randn(4, 4)
weight_2 = torch.randn(4, 4)
weight_3 = torch.randn(4, 4)

# Instantiate non-linearity
relu = nn.ReLU()

# Apply non-linearity on the hidden layers
hidden_1_activated = relu(torch.matmul(input_layer, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
print(torch.matmul(hidden_2_activated, weight_3))

# Apply non-linearity in the product of first two weights.
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))

# Multiply `weight_composed_1_activated` with `weight_3
weight = torch.matmul(weight_composed_1_activated, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))
