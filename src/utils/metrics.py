import os
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

from ..core.config import config
from ..engine.time_parser import TimeParser

def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates the given model against a dataloader.
    
    Args:
        model: The trained NLP model.
        dataloader: DataLoader containing validation/test data.
        
    Returns:
        Tuple[float, List]: Accuracy percentage and a list of mispredicted sentence details.
    """
    model.eval()
    correct_sentences = 0
    total_sentences = len(dataloader.dataset)
    errors = []
    
    with torch.no_grad():
        for _, targets, raw_words_batch in dataloader:
            batch_pred_tuples = model.predict(raw_words_batch)
            
            for i in range(len(raw_words_batch)):
                length = len(raw_words_batch[i])
                true_tags = [config.IDX2TAG[idx.item()] for idx in targets[i][:length]]
                true_tuples = TimeParser.extract_temporal_tuples(raw_words_batch[i], true_tags)
                
                if batch_pred_tuples[i] == true_tuples:
                    correct_sentences += 1
                else:
                    errors.append({
                        "sentence": " ".join(raw_words_batch[i]),
                        "predicted": batch_pred_tuples[i],
                        "true": true_tuples
                    })
                    
    accuracy = correct_sentences / total_sentences if total_sentences > 0 else 0.0
    return accuracy, errors

def plot_training_curves(train_losses: List[float], val_losses: List[float], val_accuracies: List[float], epochs: int) -> None:
    """Plots and saves the training metrics curves."""
    os.makedirs("assets/images", exist_ok=True)
    epochs_range = range(1, epochs + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('assets/images/loss_curve.png')
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='green', marker='^')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('assets/images/accuracy_curve.png')
    plt.close()