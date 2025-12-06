import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob

# ==========================================
# 1. Define the Model
# ==========================================
class TemperatureRegressionModel(nn.Module):
    def __init__(self):
        super(TemperatureRegressionModel, self).__init__()
        # Simple CNN for regression
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output: Single temperature value
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ==========================================
# 2. Define the Dataset
# ==========================================
class ThermalDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            labels_file (string): Path to the csv file with annotations (filename, temperature).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        # self.labels = pd.read_csv(labels_file) # Requires pandas
        # Mock data loading
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64)) # Resize to fixed size
        
        # Normalize to 0-1 and CHW format
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # Mock label: Extract temp from filename or load from CSV
        # temp = self.labels.iloc[idx, 1]
        temp = 37.0 # Placeholder
        
        return torch.tensor(image), torch.tensor([temp], dtype=torch.float32)

# ==========================================
# 3. Training Loop
# ==========================================
def train():
    # Hyperparameters
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 10
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = TemperatureRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Load Data (Placeholder paths)
    # dataset = ThermalDataset("data/train", "data/labels.csv")
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Starting training... (This is a skeleton script)")
    
    # Mock loop
    for epoch in range(EPOCHS):
        running_loss = 0.0
        # for images, temps in dataloader:
        #     images, temps = images.to(device), temps.to(device)
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, temps)
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "models/temp_regressor.pth")
    print("Model saved to models/temp_regressor.pth")

if __name__ == "__main__":
    train()
