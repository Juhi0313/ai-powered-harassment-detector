import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


HARASSMENT_DATASET_PATH = RAW_DATA_DIR / "harassment_dataset.csv"
MISOGYNY_DATASET_PATH = RAW_DATA_DIR / "misogyny_dataset.csv"


MODELS_DIR = BASE_DIR / "models"
HARASSMENT_MODEL_DIR = MODELS_DIR / "harassment_model"
MISOGYNY_MODEL_DIR = MODELS_DIR / "misogyny_model"


HARASSMENT_MODEL_CONFIG = {
    "base_model": "roberta-base",  # You can change to "distilroberta-base" for faster training
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "threshold": 0.5  # Classification threshold
}

MISOGYNY_MODEL_CONFIG = {
    # OPTION 1: Train from scratch with RoBERTa
    "base_model": "roberta-base",
    
    # OPTION 2: Use pre-trained misogyny model (uncomment to use)
    # "base_model": "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal",
    
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "threshold": 0.5
}


TOXICITY_WEIGHTS = {
    "harassment_weight": 0.6,
    "misogyny_weight": 0.4
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}


DATASET_COLUMNS = {
    "harassment": {
        "text_column": "text",  # Column name containing comments
        "label_column": "label"  # Column name containing labels (0/1 or True/False)
    },
    "misogyny": {
        "text_column": "text",
        "label_column": "label"
    }
}


for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  HARASSMENT_MODEL_DIR, MISOGYNY_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)