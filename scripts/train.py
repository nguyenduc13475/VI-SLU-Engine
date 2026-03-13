import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.core.config import config
from src.engine.dataset import SmartHomeDataset
from src.engine.bigru import IntentBiGRU
from src.utils.metrics import evaluate_model, plot_training_curves

def train():
    print("🚀 Initializing Training Process...")
    
    # Initialize Model
    model = IntentBiGRU()
    vocab = model.vocab
    
    # Prepare Data
    train_dataset = SmartHomeDataset(config.TRAIN_DATA_PATH, vocab)
    val_dataset = SmartHomeDataset(config.VAL_DATA_PATH, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=IntentBiGRU.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=IntentBiGRU.custom_collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=config.TAG2IDX['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    train_losses, val_losses, val_accuracies = [], [], []
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate Accuracy
        val_acc, _ = evaluate_model(model, val_loader)
        val_accuracies.append(val_acc * 100)
        
        print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc*100:.2f}%")

    # Save outputs
    plot_training_curves(train_losses, val_losses, val_accuracies, config.EPOCHS)
    
    os.makedirs(os.path.dirname(config.MODEL_WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_WEIGHTS_PATH)
    print(f"✅ Training Complete. Model saved to {config.MODEL_WEIGHTS_PATH}")

if __name__ == "__main__":
    train()