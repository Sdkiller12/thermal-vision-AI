
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path

# ==========================================
# 1. Define the Model
# ==========================================
class TemperatureRegressionModel(nn.Module):
    def __init__(self, in_channels=1):
        super(TemperatureRegressionModel, self).__init__()
        # Simple CNN for regression
        # Input: (B, 1, 64, 64) for Thermal
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
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
class FLIRThermalDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
        Args:
            root_dir (string): Root directory (e.g., 'data').
            subset (string): 'train' or 'val'.
            transform (callable, optional): Optional transform.
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.transform = transform
        
        # Path to images and annotations
        # Based on user structure: data/images_thermal_train/analyticsData
        # and data/images_thermal_train/coco.json
        self.image_dir = self.root_dir / f"images_thermal_{subset}" / "analyticsData"
        self.coco_path = self.root_dir / f"images_thermal_{subset}" / "coco.json"
        
        self.samples = []
        
        if self.coco_path.exists():
            print(f"Loading annotations from {self.coco_path}...")
            try:
                with open(self.coco_path, 'r') as f:
                    self.coco = json.load(f)
                
                # Map image_id to filename
                self.images = {img['id']: img['file_name'] for img in self.coco['images']}
                
                # Create samples from annotations (Objects)
                count = 0
                for ann in self.coco['annotations']:
                    img_id = ann['image_id']
                    if img_id in self.images:
                        fname = self.images[img_id]
                        
                        # FLIR Filename fix: "data/video-..." -> "video-..."
                        # Only basename is needed as we look in analyticsData
                        fname = os.path.basename(fname)
                        
                        # Verify file exists
                        full_path = self.image_dir / fname
                        # We won't check os.path.exists here for speed, but rely on loader
                        
                        bbox = ann['bbox'] # [x, y, w, h]
                        self.samples.append({
                            'path': full_path,
                            'bbox': bbox,
                            'category': ann['category_id']
                        })
                        count += 1
                        # Limit for user demo? No, load all.
                print(f"Found {count} annotation samples.")
            except Exception as e:
                print(f"Error parsing COCO json: {e}")
        
        # Fallback if specific annotations didn't produce samples or file missing
        if len(self.samples) == 0:
            print(f"Warning: No valid annotations found using {self.coco_path}. Falling back to all .tiff files.")
            # Fallback: Just load full images
            # Check if directory exists
            if self.image_dir.exists():
                files = list(self.image_dir.glob("*.tiff"))
                for p in files:
                    self.samples.append({
                        'path': p,
                        'bbox': None
                    })
                print(f"Found {len(self.samples)} images (no annotations).")
            else:
                print(f"Error: Image directory {self.image_dir} does not exist.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = str(sample['path'])
        
        # Load 16-bit TIFF
        # IMREAD_ANYDEPTH is crucial for 16-bit
        # IMREAD_GRAYSCALE would convert to 8-bit output which kills temp data
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        
        if image is None:
            # Fallback for error/missing file
            image = np.zeros((64, 64), dtype=np.float32)
            
        # Handle crop if bbox exists
        if sample['bbox']:
            x, y, w, h = map(int, sample['bbox'])
            # Clamp
            h_img, w_img = image.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            if w > 0 and h > 0:
                image = image[y:y+h, x:x+w]
            
        # Resize to fixed size
        try:
            image = cv2.resize(image, (64, 64))
        except Exception:
            image = np.zeros((64, 64), dtype=np.float32)
        
        # Normalization
        # Raw thermal data to float
        img_float = image.astype(np.float32)
        
        # Target: Max value (approx body temp logic)
        target_temp = np.max(img_float)
        
        # Normalize Input Image (Standardize)
        # Avoid div by zero
        mean = np.mean(img_float)
        std = np.std(img_float) + 1e-5
        img_norm = (img_float - mean) / std
        
        # Shape: (1, 64, 64)
        img_tensor = torch.tensor(img_norm).unsqueeze(0)
        
        # Target: single float
        target_tensor = torch.tensor([target_temp], dtype=torch.float32)
        
        return img_tensor, target_tensor

# ==========================================
# 3. Training Loop
# ==========================================
def train():
    # Hyperparameters
    BATCH_SIZE = 16
    LR = 0.001
    EPOCHS = 5
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    # Input channels = 1 (Thermal)
    model = TemperatureRegressionModel(in_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Load Data
    data_path = "data" 
    
    print("Initializing Dataset...")
    train_dataset = FLIRThermalDataset(data_path, subset='train')
    
    if len(train_dataset) == 0:
        print("No training data found used FLIR structure. Checking generic data folder...")
        return
        
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Starting training on {len(train_dataset)} samples...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        
        for i, (images, temps) in enumerate(dataloader):
            images, temps = images.to(device), temps.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, temps)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 0 and i > 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {epoch_loss:.4f}")

    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/thermal_regressor.pth")
    print("Model saved to models/thermal_regressor.pth")

if __name__ == "__main__":
    train()
