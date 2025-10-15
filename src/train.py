import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from model import Model  # Assuming model.py contains the Model class

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx, 0]
        response = self.data.iloc[idx, 1]
        return prompt, response

def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        for prompts, responses in dataloader:
            optimizer.zero_grad()
            outputs = model(prompts)
            loss = criterion(outputs, responses)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    # Load dataset
    dataset = CustomDataset('data/dataset.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer)

    # Save the model weights
    torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == '__main__':
    main()