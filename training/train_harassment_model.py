import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config.config import (
    HARASSMENT_MODEL_CONFIG,
    HARASSMENT_MODEL_DIR,
    HARASSMENT_DATASET_PATH
)
from data.data_loader import DataLoader as CustomDataLoader
from models.base_model import RoBERTaToxicityClassifier, ToxicityDataset
from training.utils import ModelTrainer, print_evaluation_report, save_model_metadata


def train_harassment_model():
    """
    Train the harassment detection model.
    
    DATASET REQUIREMENT:
    Place your Women harassment dataset at: data/raw/harassment_dataset.csv
    
    The dataset should have columns for:
    - text: The comment/message text
    - label: Binary label (0=not harassment, 1=harassment)
    """
    
    print("="*70)
    print("HARASSMENT DETECTION MODEL TRAINING")
    print("="*70)
    
    
    if not HARASSMENT_DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {HARASSMENT_DATASET_PATH}")
        print("\nPlease place your harassment dataset in data/raw/harassment_dataset.csv")
        print("Expected format: CSV with 'text' and 'label' columns")
        return
    
    
    print("\n[1/6] Loading harassment dataset...")
    data_loader = CustomDataLoader()
    
    try:
        df = data_loader.load_harassment_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    
    stats = data_loader.get_dataset_statistics(df)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    
    print("\n[2/6] Splitting dataset into train/test sets...")
    train_df, test_df = data_loader.prepare_train_test_split(df, test_size=0.2)
    
    
    print("\n[3/6] Initializing RoBERTa model...")
    classifier = RoBERTaToxicityClassifier(
        model_name=HARASSMENT_MODEL_CONFIG['base_model'],
        max_length=HARASSMENT_MODEL_CONFIG['max_length']
    )
    
    model = classifier.create_model(num_labels=2)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    
    print("\n[4/6] Creating PyTorch datasets...")
    train_dataset = ToxicityDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=HARASSMENT_MODEL_CONFIG['max_length']
    )
    
    test_dataset = ToxicityDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=HARASSMENT_MODEL_CONFIG['max_length']
    )
    
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=HARASSMENT_MODEL_CONFIG['batch_size'],
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=HARASSMENT_MODEL_CONFIG['batch_size'],
        shuffle=False
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    
    print("\n[5/6] Training model...")
    trainer = ModelTrainer(
        model=model,
        device=classifier.device,
        config=HARASSMENT_MODEL_CONFIG
    )
    
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        save_path=str(HARASSMENT_MODEL_DIR)
    )
    
   
    print("\n[6/6] Evaluating and saving model...")
    final_metrics = trainer.evaluate(test_dataloader)
    print_evaluation_report(final_metrics, "Harassment Detection Model")
    
    
    classifier.model = model
    classifier.save_model(str(HARASSMENT_MODEL_DIR))
    
    
    dataset_info = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'positive_ratio': stats['positive_ratio'],
        'source': str(HARASSMENT_DATASET_PATH)
    }
    save_model_metadata(str(HARASSMENT_MODEL_DIR), final_metrics, dataset_info)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {HARASSMENT_MODEL_DIR}")
    print(f"Best Validation Accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Final F1 Score: {final_metrics['f1']:.4f}")
    print(f"Final ROC-AUC: {final_metrics['roc_auc']:.4f}")
    
    
    print("\n" + "="*70)
    print("TESTING MODEL PREDICTIONS")
    print("="*70)
    
    test_texts = [
        "You're such a stupid woman, go back to the kitchen",
        "Great work on the project! Very impressive results.",
        "Women shouldn't be allowed to work",
        "Thank you for your contribution to the discussion"
    ]
    
    print("\nSample Predictions:")
    for text in test_texts:
        result = classifier.predict_single(text)
        label = "HARASSMENT" if result['prediction'] == 1 else "NOT HARASSMENT"
        score = result['toxicity_score']
        print(f"\nText: {text}")
        print(f"Prediction: {label}")
        print(f"Toxicity Score: {score:.4f}")


if __name__ == "__main__":
    train_harassment_model()