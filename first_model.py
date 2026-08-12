import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Training data: y = 2x
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Simple neural network with one input and one output
model = nn.Linear(1, 1)

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

print("\nStarting training...")

for epoch in range(1000):

    prediction = model(x)

    loss = loss_function(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

print("Training complete.")
print("Final loss:", loss.item())

# Test the trained model
test_value = torch.tensor([[5.0]])

with torch.no_grad():
    result = model(test_value)

print("\nInput: 5")
print("Expected result: 10")
print("Model prediction:", result.item())