import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple, Dict

from ..core.config import config
from .time_parser import TimeParser

def load_w2v(path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Loads custom Word2Vec embeddings from a text/vec file.
    
    Args:
        path (str): Path to the .vec file.
        
    Returns:
        Tuple[Dict[str, int], np.ndarray]: The vocabulary dictionary and the embedding matrix.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1, "TIME_TOKEN": 2}
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Word2Vec file not found at {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        embed_dim = int(lines[0].strip().split()[1])
        embeds = [
            np.zeros(embed_dim),  # <PAD> vector
            np.random.uniform(-0.1, 0.1, embed_dim),  # <UNK> vector
            np.random.uniform(-0.1, 0.1, embed_dim)   # TIME_TOKEN vector
        ]
        for line in lines[1:]:
            parts = line.strip().split()
            word = parts[0]
            if word not in vocab:
                vec = np.array([float(x) for x in parts[1:]])
                vocab[word] = len(vocab)
                embeds.append(vec)
                
    return vocab, np.array(embeds)


class IntentBiGRU(nn.Module):
    """
    Bidirectional GRU Model for Intent and Temporal Slot Extraction.
    """
    
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        
        # Load Vocabulary and Embeddings
        self.vocab, embedding_matrix = load_w2v(config.W2V_PATH)
        self.vocab_size = len(self.vocab)
        self.embed_dim = embedding_matrix.shape[1]
        
        # Architecture
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        self.gru = nn.GRU(
            input_size=self.embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, config.NUM_CLASSES)

        self.device = config.get_device()
        self.to(self.device)
        self.load_weights()

    def load_weights(self) -> None:
        """Loads pre-trained weights if available."""
        if os.path.exists(config.MODEL_WEIGHTS_PATH):
            self.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=self.device))
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BiGRU model."""
        emb = self.embedding(x)
        gru_out, _ = self.gru(emb) 
        x = self.dropout(gru_out)
        out = self.fc(x)
        return out
    
    def predict(self, texts: Union[str, List[str], List[List[str]]]) -> Union[List[Tuple], List[List[Tuple]]]:
        """
        Predicts semantic tuples for given text(s).
        """
        self.eval()

        if isinstance(texts, str):
            raw_batch = [texts.strip().split()]
            is_single = True
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], str):
            raw_batch = [t.strip().split() for t in texts]
            is_single = False
        else:
            raw_batch = texts
            is_single = False
            
        UNK_IDX = self.vocab.get('<UNK>', 1) 
        TIME_TOKEN_IDX = self.vocab.get('TIME_TOKEN', 2)
        PAD_IDX = 0
        
        max_len = max(len(words) for words in raw_batch)
        batch_indices = []
        
        for words in raw_batch:
            indices = []
            for w in words:
                if TimeParser.is_time_word(w):
                    indices.append(TIME_TOKEN_IDX)
                else:
                    indices.append(self.vocab.get(w, UNK_IDX))
            indices += [PAD_IDX] * (max_len - len(indices))
            batch_indices.append(indices)
            
        inputs = torch.tensor(batch_indices, dtype=torch.long).to(self.device)
        batch_size = inputs.size(0)
        
        with torch.no_grad():
            outputs = self(inputs)
            
            # Masking logic to force temporal tags on time tokens
            time_tag_idx = torch.tensor([
                config.TAG2IDX[t] for t in ['TimeAfter', 'TimeWithin', 'TimeRepeat', 'TimeRange']
            ], device=self.device)
            
            is_time_token = (inputs == TIME_TOKEN_IDX).unsqueeze(-1) 
            is_time_tag_class = torch.zeros(config.NUM_CLASSES, dtype=torch.bool, device=self.device)
            is_time_tag_class[time_tag_idx] = True
            
            outputs.masked_fill_((~is_time_token) & is_time_tag_class, -1e9)
            outputs.masked_fill_(is_time_token & (~is_time_tag_class), -1e9)
            
            # Logit Pooling for consecutive time tokens
            for b in range(batch_size):
                length = len(raw_batch[b])
                i = 0
                while i < length:
                    if inputs[b, i] == TIME_TOKEN_IDX: 
                        start = i
                        while i < length and inputs[b, i] == TIME_TOKEN_IDX:
                            i += 1
                        end = i
                        pooled_logits = outputs[b, start:end].sum(dim=0, keepdim=True) 
                        outputs[b, start:end] = pooled_logits 
                    else:
                        i += 1

            preds = torch.argmax(outputs, dim=-1)
            
        batch_results = []
        for b in range(batch_size):
            length = len(raw_batch[b])
            pred_tags = [config.IDX2TAG[idx.item()] for idx in preds[b][:length]]
            pred_tuples = TimeParser.extract_temporal_tuples(raw_batch[b], pred_tags)
            batch_results.append(pred_tuples)
        
        return batch_results[0] if is_single else batch_results

    @staticmethod
    def custom_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """Custom collate function for DataLoader to handle raw words."""
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        raw_words = [item[2] for item in batch]
        return inputs, targets, raw_words