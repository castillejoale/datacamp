# Import torch
import torch

# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Multiply x with y
q = torch.matmul(x,y)

# Multiply elementwise z with q
f = q*z

mean_f = torch.mean(f)
print(mean_f)
