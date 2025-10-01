import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np
from typing import Dict, Tuple
import json
from pathlib import Path

class ModelTrainer:
    """Utility class for training toxicity detection models."""
    
    def __init__(self, model, device, config: Dict):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            device: torch device
            config: Training configuration dictionary
        """
        self.model = model
        self.device = device
        self.config = config
        
      
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
    def create_optimizer_and_scheduler(self, train_dataloader):
        """Create optimizer and learning rate scheduler."""
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        
        total_steps = len(train_dataloader) * self.config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_dataloader, optimizer, scheduler) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        return avg_loss
    
    def evaluate(self, val_dataloader) -> Dict:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        
        avg_loss = total_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        
        try:
            all_probabilities = np.array(all_probabilities)
            roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except:
            roc_auc = 0.0
        
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def train(self, train_dataloader, val_dataloader, save_path: str) -> Dict:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        optimizer, scheduler = self.create_optimizer_and_scheduler(train_dataloader)
        
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        print(f"Total training steps: {len(train_dataloader) * self.config['num_epochs']}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            
            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler)
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")
            
          
            val_metrics = self.evaluate(val_dataloader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            print(f"\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  F1 Score: {val_metrics['f1']:.4f}")
            print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
            
            
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                print(f"\n  New best model! Accuracy: {self.best_val_accuracy:.4f}")
        
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with accuracy: {self.best_val_accuracy:.4f}")
        
       
        final_metrics = self.evaluate(val_dataloader)
        
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'final_metrics': final_metrics,
            'config': self.config
        }
        
        
        history_path = Path(save_path).parent / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining history saved to: {history_path}")
        
        return history


def print_evaluation_report(metrics: Dict, model_type: str = "Model"):
    """Print a detailed evaluation report."""
    print(f"\n{'='*60}")
    print(f"{model_type} Evaluation Report")
    print(f"{'='*60}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    
    if cm[0][1] + cm[0][0] > 0:
        false_positive_rate = cm[0][1] / (cm[0][1] + cm[0][0])
        print(f"\n  False Positive Rate: {false_positive_rate:.4f}")
    
    if cm[1][0] + cm[1][1] > 0:
        false_negative_rate = cm[1][0] / (cm[1][0] + cm[1][1])
        print(f"  False Negative Rate: {false_negative_rate:.4f}")
    
    print(f"\n{'='*60}\n")


def save_model_metadata(save_path: str, metrics: Dict, dataset_info: Dict):
    """Save model metadata including performance metrics and dataset info."""
    metadata = {
        'performance_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'roc_auc': float(metrics['roc_auc'])
        },
        'dataset_info': dataset_info,
        'confusion_matrix': metrics['confusion_matrix'],
        'timestamp': str(np.datetime64('now'))
    }
    
    metadata_path = Path(save_path) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to: {metadata_path}")