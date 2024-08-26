import torch
import torch.optim as optim
from src.models import MAML_GNN_Model

# Training parameters
num_epochs = 10
learning_rate = 0.001

# Load data (dummy example, replace with actual data)
data_loader = ...  # Replace with your DataLoader
n_classes = 3  # Number of classes

# Initialize model and optimizer
model = MAML_GNN_Model(in_channels=7, out_channels=n_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    for data in data_loader:
        ct_data, pet_data, fused_data, labels = data
        optimizer.zero_grad()
        output = model(ct_data, pet_data, fused_data)
        loss = torch.nn.functional.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
