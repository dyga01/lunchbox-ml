import os
import torch
import torch.nn as nn
import numpy as np
import time
from data_preprocessing import preprocess_data

# Path to the dataset
data_path = "Metro_total_monthly_payment_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
# Preprocess the data
X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(data_path, sequence_length=10)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the Bidirectional GRU model
class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(BidirectionalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Model parameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 1

# Initialize the model, loss function, and optimizer
model = BidirectionalGRU(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate

print("\nTraining The Bidirectional GRU Model\n")

# Training loop
num_epochs = 10
train_losses = []
val_losses = []

# Start timing
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i in range(len(X_train)):
        outputs = model(X_train[i:i+1])  # Process one sample at a time
        loss = criterion(outputs, y_train[i:i+1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(X_train)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(len(X_test)):
            outputs = model(X_test[i:i+1])
            loss = criterion(outputs, y_test[i:i+1])
            val_loss += loss.item()

    val_loss /= len(X_test)
    val_losses.append(val_loss)

    # Log the losses
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")

# End timing
end_time = time.time()
training_time = end_time - start_time
print(f"\nTotal Training Time: {training_time:.2f} seconds")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f"Test Loss: {test_loss.item():.8f}")
    # Convert PyTorch tensors to NumPy arrays for scaling
    predictions_np = predictions.detach().numpy()
    y_test_np = y_test.detach().numpy()
    # Inverse transform to get actual values
    predictions_actual = scaler_y.inverse_transform(predictions_np)
    y_test_actual = scaler_y.inverse_transform(y_test_np)
    # Calculate additional metrics if desired
    rmse = np.sqrt(((predictions_actual - y_test_actual) ** 2).mean())
    print(f"Root Mean Square Error (RMSE): ${rmse:.2f}")
