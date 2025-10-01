import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import sys
sys.path.append('..')
from config.config import (
    HARASSMENT_DATASET_PATH, 
    MISOGYNY_DATASET_PATH,
    DATASET_COLUMNS
)

class DataLoader:
    """
    Loads and preprocesses datasets for training harassment and misogyny detection models.
    
    IMPORTANT: Place your datasets in data/raw/ directory:
    - harassment_dataset.csv: Women harassment dataset
    - misogyny_dataset.csv: Your misogyny dataset or download from Hugging Face
    """
    
    def __init__(self):
        self.harassment_data = None
        self.misogyny_data = None
    
    def load_harassment_dataset(self) -> pd.DataFrame:
        """
        Load harassment dataset.
        
        DATASET LINK: Place your Women harassment dataset at:
        data/raw/harassment_dataset.csv
        
        Expected columns (adjust in config.py if different):
        - text: The comment text
        - label: Binary label (0=not harassment, 1=harassment)
        """
        try:
            df = pd.read_csv(HARASSMENT_DATASET_PATH)
            print(f"Loaded harassment dataset: {len(df)} samples")
            print(f"Columns: {df.columns.tolist()}")
            
            
            text_col = DATASET_COLUMNS["harassment"]["text_column"]
            label_col = DATASET_COLUMNS["harassment"]["label_column"]
            
            
            if text_col not in df.columns or label_col not in df.columns:
                raise ValueError(
                    f"Expected columns '{text_col}' and '{label_col}' not found. "
                    f"Available columns: {df.columns.tolist()}. "
                    f"Update DATASET_COLUMNS in config.py"
                )
            
            
            df = df[[text_col, label_col]].rename(
                columns={text_col: "text", label_col: "label"}
            )
            
            
            df = self._clean_dataset(df)
            self.harassment_data = df
            
            return df
        
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Harassment dataset not found at {HARASSMENT_DATASET_PATH}. "
                "Please place your dataset in data/raw/harassment_dataset.csv"
            )
    
    def load_misogyny_dataset(self) -> pd.DataFrame:
        """
        Load misogyny dataset.
        
        DATASET LINK: You can either:
        1. Place your own dataset at data/raw/misogyny_dataset.csv
        2. Use the Hugging Face dataset: annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal
           (Download and convert to CSV format)
        
        Expected columns (adjust in config.py if different):
        - text: The comment text
        - label: Binary label (0=not misogynistic, 1=misogynistic)
        """
        try:
            df = pd.read_csv(MISOGYNY_DATASET_PATH)
            print(f"Loaded misogyny dataset: {len(df)} samples")
            print(f"Columns: {df.columns.tolist()}")
            
            
            text_col = DATASET_COLUMNS["misogyny"]["text_column"]
            label_col = DATASET_COLUMNS["misogyny"]["label_column"]
            
           
            if text_col not in df.columns or label_col not in df.columns:
                raise ValueError(
                    f"Expected columns '{text_col}' and '{label_col}' not found. "
                    f"Available columns: {df.columns.tolist()}. "
                    f"Update DATASET_COLUMNS in config.py"
                )
            
            
            df = df[[text_col, label_col]].rename(
                columns={text_col: "text", label_col: "label"}
            )
            
         
            df = self._clean_dataset(df)
            self.misogyny_data = df
            
            return df
        
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Misogyny dataset not found at {MISOGYNY_DATASET_PATH}. "
                "Please place your dataset in data/raw/misogyny_dataset.csv or "
                "download from: annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal"
            )
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess dataset."""
       
        df = df.drop_duplicates(subset=['text'])
        
        
        df = df.dropna(subset=['text', 'label'])
        
       
        df['text'] = df['text'].astype(str).str.strip()
        
        df = df[df['text'].str.len() > 0]
        
        
        df['label'] = df['label'].astype(int)
        
        
        unique_labels = df['label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            print(f"Warning: Found non-binary labels: {unique_labels}. Converting to binary.")
            df['label'] = (df['label'] > 0).astype(int)
        
        print(f"After cleaning: {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def prepare_train_test_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets."""
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']  # Maintain label distribution
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Get basic statistics about the dataset."""
        stats = {
            "total_samples": len(df),
            "positive_samples": (df['label'] == 1).sum(),
            "negative_samples": (df['label'] == 0).sum(),
            "positive_ratio": (df['label'] == 1).mean(),
            "avg_text_length": df['text'].str.len().mean(),
            "max_text_length": df['text'].str.len().max(),
            "min_text_length": df['text'].str.len().min()
        }
        return stats



def download_huggingface_misogyny_dataset(save_path: str = None):
    """
    Helper function to download misogyny dataset from Hugging Face.
    
    Usage:
    from data.data_loader import download_huggingface_misogyny_dataset
    download_huggingface_misogyny_dataset()
    """
    try:
        from datasets import load_dataset
        
        print("Downloading misogyny dataset from Hugging Face...")
        print("Dataset: annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal")
        
        # Load dataset
        dataset = load_dataset("annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal")
        
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test']) if 'test' in dataset else None
        
        
        if test_df is not None:
            df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            df = train_df
        
        if save_path is None:
            save_path = MISOGYNY_DATASET_PATH
        
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to: {save_path}")
        print(f"Total samples: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download manually or provide your own dataset.")
        return None


if __name__ == "__main__":
    
    loader = DataLoader()
    
    print("\n=== Testing Harassment Dataset ===")
    try:
        harassment_df = loader.load_harassment_dataset()
        stats = loader.get_dataset_statistics(harassment_df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Testing Misogyny Dataset ===")
    try:
        misogyny_df = loader.load_misogyny_dataset()
        stats = loader.get_dataset_statistics(misogyny_df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")