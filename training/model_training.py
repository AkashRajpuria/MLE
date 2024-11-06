import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
import logging

# Configure logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IrisClassifier(nn.Module):
    """A neural network model for classifying Iris flower types."""
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Linear(4, 10)
        self.layer2 = nn.Linear(10, 6)
        self.layer3 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Defines the forward pass through the network."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

def prepare_data_loader():
    """Loads data, processes it into tensors, and returns a DataLoader for training."""
    try:
        # Load the data
        train_df = pd.read_csv('data/training.csv')
        X_train = train_df.drop('target', axis=1).values
        y_train = train_df['target'].values

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        return train_loader
    except FileNotFoundError:
        logging.error("Training data file not found. Ensure 'data/training.csv' exists.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in data loading: {e}")
        raise

def execute_training():
    """Trains the IrisClassifier model, saving it upon completion."""
    try:
        start_time = time.time()

        # Load data
        train_loader = prepare_data_loader()
        
        # Initialize model, criterion, and optimizer
        model = IrisClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Logging loss for each epoch
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
        
        # Save the trained model
        torch.save(model.state_dict(), 'model.pth')
        logging.info(f'Training completed in {time.time() - start_time:.2f} seconds')
        
    except Exception as e:
        logging.error(f"Error during training: {e}")

if __name__ == "__main__":
    execute_training()
