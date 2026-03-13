import os
from typing import List, Dict
import torch

class AppConfig:
    """
    Core configuration settings for the Vi-SLU Engine.
    This class centralizes all hyperparameters, vocabulary mappings, and system paths.
    """
    
    # ---------------------------------------------------------
    # 1. FILE PATHS
    # ---------------------------------------------------------
    W2V_PATH: str = os.getenv("W2V_PATH", "models/mini_phow2v_100d.vec")
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "models/bigru_model.pth")
    TRAIN_DATA_PATH: str = os.getenv("TRAIN_DATA_PATH", "data/vi_smarthome_commands_train.csv")
    VAL_DATA_PATH: str = os.getenv("VAL_DATA_PATH", "data/vi_smarthome_commands_validation.csv")

    # ---------------------------------------------------------
    # 2. MODEL HYPERPARAMETERS
    # ---------------------------------------------------------
    EMBED_DIM: int = 100
    MAX_LEN: int = 30
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LR: float = 0.001

    # ---------------------------------------------------------
    # 3. TAGS & VOCABULARY DEFINITION
    # ---------------------------------------------------------
    TAGS: List[str] = [
        '<PAD>', 'BatDen', 'TatDen', 'MoCua', 'DongCua', 'BatQuat', 'TatQuat', 
        'QuatNhanh', 'QuatCham', 'NhietDo', 'DoAm', 
        'TimeAfter', 'TimeWithin', 'TimeRepeat', 'TimeRange', 'Sep', 'O'
    ]

    NUM_CLASSES: int = len(TAGS)
    
    # Dictionary mappings for fast O(1) lookups
    TAG2IDX: Dict[str, int] = {tag: idx for idx, tag in enumerate(TAGS)}
    IDX2TAG: Dict[int, str] = {idx: tag for tag, idx in TAG2IDX.items()}

    # ---------------------------------------------------------
    # 4. HARDWARE CONFIGURATION
    # ---------------------------------------------------------
    @staticmethod
    def get_device() -> torch.device:
        """
        Dynamically allocates the processing device (GPU if available, otherwise CPU).
        
        Returns:
            torch.device: The designated computation device.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a singleton configuration object to be imported across the app
config = AppConfig()