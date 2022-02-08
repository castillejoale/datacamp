# Import torch
import torch
import torch.nn as nn
torch.manual_seed(314)

input_layer = torch.randn(1, 4)
weight_1 = torch.randn(4, 4)
weight_2 = torch.randn(4, 4)
weight_3 = torch.randn(4, 4)

# Calculate the first and second hidden layer
hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)

# Calculate the output
print(torch.matmul(hidden_2, weight_3))

# Calculate weight_composed_1 and weight
weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))
