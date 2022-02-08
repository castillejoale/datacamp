# Import torch
import torch

# Initialize tensors x, y and z
x = torch.randn((1000, 1000), requires_grad=True)
y = torch.randn((1000, 1000), requires_grad=True)
z = torch.randn((1000, 1000), requires_grad=True)

# Multiply tensors x and y
q = torch.matmul(x,y)

# Elementwise multiply tensors z with q
f = z * q

mean_f = torch.mean(f)

# Print mean_f
print(str(mean_f))

# Calculate the gradients
mean_f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))
