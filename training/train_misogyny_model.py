import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config.config import (
    MISOGYNY_MODEL_CONFIG,
    MISOGYNY_MODEL_DIR,
    MISOGYNY_DATASET_PATH
)
from data.data_loader import DataLoader as CustomDataLoader
from models.base_model import RoBERTaToxicityClassifier, ToxicityDataset
from training.utils import ModelTrainer, print_evaluation_report, save_model_metadata


def train_misogyny_model():
    """
    Train the misogyny detection model.
    
    DATASET REQUIREMENT:
    Option 1: Place your own misogyny dataset at: data/raw/misogyny_dataset.csv
    Option 2: Download from Hugging Face: annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal
    
    The dataset should have columns for:
    - text: The comment/message text
    - label: Binary label (0=not misogynistic, 1=misogynistic)
    
    You can also use the pre-trained model directly by setting:
    MISOGYNY_MODEL_CONFIG['base_model'] = "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal"
    in config.py
    """
    
    print("="*70)
    print("MISOGYNY DETECTION MODEL TRAINING")
    print("="*70)
    
   
    if not MISOGYNY_DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {MISOGYNY_DATASET_PATH}")
        print("\nPlease either:")
        print("1. Place your misogyny dataset in data/raw/misogyny_dataset.csv")
        print("2. Download from Hugging Face: annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal")
        print("\nTo download from Hugging Face, run:")
        print("  from data.data_loader import download_huggingface_misogyny_dataset")
        print("  download_huggingface_misogyny_dataset()")
        return
    
    
    print("\n[1/6] Loading misogyny dataset...")
    data_loader = CustomDataLoader()
    
    try:
        df = data_loader.load_misogyny_dataset()
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
    print(f"Base model: {MISOGYNY_MODEL_CONFIG['base_model']}")
    
    classifier = RoBERTaToxicityClassifier(
        model_name=MISOGYNY_MODEL_CONFIG['base_model'],
        max_length=MISOGYNY_MODEL_CONFIG['max_length']
    )
    
    model = classifier.create_model(num_labels=2)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    
    print("\n[4/6] Creating PyTorch datasets...")
    train_dataset = ToxicityDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=MISOGYNY_MODEL_CONFIG['max_length']
    )
    
    test_dataset = ToxicityDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=MISOGYNY_MODEL_CONFIG['max_length']
    )
    
   
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=MISOGYNY_MODEL_CONFIG['batch_size'],
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=MISOGYNY_MODEL_CONFIG['batch_size'],
        shuffle=False
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    print("\n[5/6] Training model...")
    trainer = ModelTrainer(
        model=model,
        device=classifier.device,
        config=MISOGYNY_MODEL_CONFIG
    )
    
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        save_path=str(MISOGYNY_MODEL_DIR)
    )
    
    print("\n[6/6] Evaluating and saving model...")
    final_metrics = trainer.evaluate(test_dataloader)
    print_evaluation_report(final_metrics, "Misogyny Detection Model")
    
   
    classifier.model = model
    classifier.save_model(str(MISOGYNY_MODEL_DIR))
    
    
    dataset_info = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'positive_ratio': stats['positive_ratio'],
        'source': str(MISOGYNY_DATASET_PATH)
    }
    save_model_metadata(str(MISOGYNY_MODEL_DIR), final_metrics, dataset_info)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {MISOGYNY_MODEL_DIR}")
    print(f"Best Validation Accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Final F1 Score: {final_metrics['f1']:.4f}")
    print(f"Final ROC-AUC: {final_metrics['roc_auc']:.4f}")
    
    
    print("\n" + "="*70)
    print("TESTING MODEL PREDICTIONS")
    print("="*70)
    
    test_texts = [
        "Women are inferior and shouldn't have rights",
        "Congratulations on your achievement!",
        "She's just an emotional female, can't trust her judgment",
        "Both men and women bring valuable perspectives to the team"
    ]
    
    print("\nSample Predictions:")
    for text in test_texts:
        result = classifier.predict_single(text)
        label = "MISOGYNISTIC" if result['prediction'] == 1 else "NOT MISOGYNISTIC"
        score = result['toxicity_score']
        print(f"\nText: {text}")
        print(f"Prediction: {label}")
        print(f"Toxicity Score: {score:.4f}")


if __name__ == "__main__":
    train_misogyny_model()