import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from ..core.config import config

class SmartHomeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing the Smart Home CSV data.
    Handles padding and vocabulary mapping dynamically.
    """
    
    def __init__(self, csv_file: str, vocab: Dict[str, int]) -> None:
        """
        Initializes the dataset.
        
        Args:
            csv_file (str): Path to the training or validation CSV file.
            vocab (Dict[str, int]): Loaded word-to-index mapping vocabulary.
        """
        self.sentences: List[torch.Tensor] = []
        self.tags: List[torch.Tensor] = []
        self.raw_words: List[List[str]] = []
        self.vocab = vocab
        
        self._prepare_data(csv_file)

    def _is_time_word(self, word: str) -> bool:
        """Helper to identify temporal words (can be imported instead)."""
        if str(word).isdigit(): return True
        return str(word).lower() in ['ngày', 'giây', 'phút', 'giờ', 'tiếng', 'rưỡi', 'nửa']

    def _prepare_data(self, csv_file: str) -> None:
        """Reads CSV and converts words/tags to padded tensors."""
        df = pd.read_csv(csv_file)
        
        for _, group in df.groupby('sentence'):
            words = group['word'].tolist()
            tgs = group['tag'].tolist()
            self.raw_words.append(words)
            
            # Map word to index, handling OOV (Out-Of-Vocabulary) and TIME_TOKEN
            w_idx = []
            for w in words:
                if self._is_time_word(w):
                    w_idx.append(self.vocab.get("TIME_TOKEN", 2))
                else:
                    w_idx.append(self.vocab.get(w, self.vocab.get("<UNK>", 1)))
                    
            t_idx = [config.TAG2IDX.get(t, 0) for t in tgs]
            
            # Padding
            if len(w_idx) < config.MAX_LEN:
                w_idx.extend([self.vocab.get("<PAD>", 0)] * (config.MAX_LEN - len(w_idx)))
                t_idx.extend([config.TAG2IDX.get("<PAD>", 0)] * (config.MAX_LEN - len(t_idx)))
            else:
                w_idx = w_idx[:config.MAX_LEN]
                t_idx = t_idx[:config.MAX_LEN]
                
            self.sentences.append(torch.tensor(w_idx, dtype=torch.long))
            self.tags.append(torch.tensor(t_idx, dtype=torch.long))
            
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        return self.sentences[idx], self.tags[idx], self.raw_words[idx]