# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import MultimodalModel
from dataset import EmotionDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- Configuration ---
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 7 # We excluded 'calm'
MODEL_SAVE_PATH = 'saved_models/best_model.pth'

def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # --- Data Loading ---
    full_dataset = EmotionDataset('processed_data/metadata.csv')
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model, Optimizer, Loss ---
    model = MultimodalModel(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []
        
        for video, audio, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            video, audio, labels = video.to(device), audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(video, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_accuracy = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch+1} Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for video, audio, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                video, audio, labels = video.to(device), audio.to(device), labels.to(device)
                
                outputs = model(video, audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved with accuracy: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    train_model()