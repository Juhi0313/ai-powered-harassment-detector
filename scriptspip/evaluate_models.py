import sys
sys.path.append('..')

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from api.predictor import ToxicityPredictor
from data.data_loader import DataLoader
from config.config import HARASSMENT_MODEL_DIR, MISOGYNY_MODEL_DIR


def evaluate_models_on_datasets():
    """
    Evaluate both models on their respective test datasets.
    
    DATASET REQUIREMENT:
    Ensure both datasets are available:
    - data/raw/harassment_dataset.csv
    - data/raw/misogyny_dataset.csv
    """
    print("="*70)
    print("MODEL EVALUATION SCRIPT")
    print("="*70)
    
    
    if not HARASSMENT_MODEL_DIR.exists():
        print(f"\nERROR: Harassment model not found at {HARASSMENT_MODEL_DIR}")
        print("Train the model first: python training/train_harassment_model.py")
        return
    
    if not MISOGYNY_MODEL_DIR.exists():
        print(f"\nERROR: Misogyny model not found at {MISOGYNY_MODEL_DIR}")
        print("Train the model first: python training/train_misogyny_model.py")
        return
    
   
    print("\n[1/4] Loading models...")
    predictor = ToxicityPredictor()
    predictor.load_models()
    
    
    print("\n[2/4] Loading datasets...")
    data_loader = DataLoader()
    
    try:
        harassment_df = data_loader.load_harassment_dataset()
        misogyny_df = data_loader.load_misogyny_dataset()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
   
    print("\n[3/4] Running predictions on sample data...")
    
    
    sample_size = min(100, len(harassment_df))
    harassment_sample = harassment_df.sample(n=sample_size, random_state=42)
    misogyny_sample = misogyny_df.sample(n=sample_size, random_state=42)
    
    print(f"\nEvaluating on {sample_size} harassment samples...")
    harassment_results = []
    for _, row in harassment_sample.iterrows():
        result = predictor.predict_single(row['text'])
        result['actual_label'] = row['label']
        harassment_results.append(result)
    
    print(f"Evaluating on {sample_size} misogyny samples...")
    misogyny_results = []
    for _, row in misogyny_sample.iterrows():
        result = predictor.predict_single(row['text'])
        result['actual_label'] = row['label']
        misogyny_results.append(result)
    
    
    print("\n[4/4] Calculating evaluation metrics...")
    
    
    harassment_correct = sum(
        1 for r in harassment_results 
        if (r['is_harassment'] and r['actual_label'] == 1) or 
           (not r['is_harassment'] and r['actual_label'] == 0)
    )
    harassment_accuracy = harassment_correct / len(harassment_results)
    
    # Misogyny metrics
    misogyny_correct = sum(
        1 for r in misogyny_results 
        if (r['is_misogyny'] and r['actual_label'] == 1) or 
           (not r['is_misogyny'] and r['actual_label'] == 0)
    )
    misogyny_accuracy = misogyny_correct / len(misogyny_results)
    
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nHarassment Model:")
    print(f"  Sample Size: {len(harassment_results)}")
    print(f"  Accuracy: {harassment_accuracy:.4f}")
    print(f"  Correct Predictions: {harassment_correct}/{len(harassment_results)}")
    
    print(f"\nMisogyny Model:")
    print(f"  Sample Size: {len(misogyny_results)}")
    print(f"  Accuracy: {misogyny_accuracy:.4f}")
    print(f"  Correct Predictions: {misogyny_correct}/{len(misogyny_results)}")
    
    
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    print("\nHarassment Detection Examples:")
    for i, result in enumerate(harassment_results[:5]):
        actual = "HARASSMENT" if result['actual_label'] == 1 else "NOT HARASSMENT"
        predicted = "HARASSMENT" if result['is_harassment'] else "NOT HARASSMENT"
        correct = "✓" if (result['is_harassment'] and result['actual_label'] == 1) or \
                         (not result['is_harassment'] and result['actual_label'] == 0) else "✗"
        
        print(f"\n{i+1}. {correct} Text: {result['text'][:70]}...")
        print(f"   Actual: {actual} | Predicted: {predicted}")
        print(f"   Score: {result['harassment_score']:.4f}")
    
    print("\nMisogyny Detection Examples:")
    for i, result in enumerate(misogyny_results[:5]):
        actual = "MISOGYNISTIC" if result['actual_label'] == 1 else "NOT MISOGYNISTIC"
        predicted = "MISOGYNISTIC" if result['is_misogyny'] else "NOT MISOGYNISTIC"
        correct = "✓" if (result['is_misogyny'] and result['actual_label'] == 1) or \
                         (not result['is_misogyny'] and result['actual_label'] == 0) else "✗"
        
        print(f"\n{i+1}. {correct} Text: {result['text'][:70]}...")
        print(f"   Actual: {actual} | Predicted: {predicted}")
        print(f"   Score: {result['misogyny_score']:.4f}")
    
    
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    
    evaluation_summary = {
        "harassment_model": {
            "sample_size": len(harassment_results),
            "accuracy": harassment_accuracy,
            "correct_predictions": harassment_correct
        },
        "misogyny_model": {
            "sample_size": len(misogyny_results),
            "accuracy": misogyny_accuracy,
            "correct_predictions": misogyny_correct
        }
    }
    
    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\n\nEvaluation results saved to: {results_dir}/evaluation_summary.json")
    print("="*70)


def test_edge_cases():
    """Test the model on various edge cases and challenging examples."""
    print("\n" + "="*70)
    print("EDGE CASE TESTING")
    print("="*70)
    
    predictor = ToxicityPredictor()
    predictor.load_models()
    
    
    edge_cases = {
        "Empty/Short": [
            "",
            "ok",
            "yes"
        ],
        "Neutral": [
            "The weather is nice today",
            "I had pasta for lunch",
            "Meeting at 3pm"
        ],
        "Sarcasm/Irony": [
            "Oh great, another mansplaining session",
            "Yeah, because women are sooo good at math",
            "Sure, let the woman handle it, what could go wrong"
        ],
        "Borderline": [
            "You're acting very emotional right now",
            "That's not very ladylike behavior",
            "Maybe you should calm down"
        ],
        "Clearly Toxic": [
            "Women are inferior and shouldn't vote",
            "Shut up you stupid woman",
            "Get back to the kitchen"
        ],
        "Context-Dependent": [
            "You throw like a girl",
            "Man up and deal with it",
            "Don't be such a drama queen"
        ]
    }
    
    for category, texts in edge_cases.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for text in texts:
            if not text:
                text = "[empty string]"
                continue
            
            result = predictor.predict_single(text)
            print(f"\nText: {text}")
            print(f"  Combined Score: {result['combined_toxicity_score']:.4f}")
            print(f"  Harassment: {result['is_harassment']} ({result['harassment_score']:.4f})")
            print(f"  Misogyny: {result['is_misogyny']} ({result['misogyny_score']:.4f})")
            print(f"  Risk Level: {result['risk_level'].upper()}")


def compare_models_performance():
    """Compare performance of harassment vs misogyny models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    predictor = ToxicityPredictor()
    predictor.load_models()
    
    
    test_cases = [
        ("You're stupid", "general_insult"),
        ("Women are stupid", "gendered_insult"),
        ("Get out of here", "harassment"),
        ("Women should stay home", "misogyny"),
        ("You're annoying", "general_insult"),
        ("Typical woman behavior", "misogyny"),
        ("I'll make your life hell", "harassment"),
        ("Women can't lead", "misogyny")
    ]
    
    print("\nAnalyzing overlap between harassment and misogyny detection:\n")
    
    results = []
    for text, category in test_cases:
        result = predictor.predict_single(text)
        results.append({
            'text': text,
            'category': category,
            'harassment_score': result['harassment_score'],
            'misogyny_score': result['misogyny_score'],
            'difference': abs(result['harassment_score'] - result['misogyny_score'])
        })
        
        print(f"Text: {text}")
        print(f"  Category: {category}")
        print(f"  Harassment Score: {result['harassment_score']:.4f}")
        print(f"  Misogyny Score:   {result['misogyny_score']:.4f}")
        print(f"  Difference:       {abs(result['harassment_score'] - result['misogyny_score']):.4f}")
        print()
    
    
    df = pd.DataFrame(results)
    print("\nAverage score difference by category:")
    avg_diff = df.groupby('category')['difference'].mean()
    for cat, diff in avg_diff.items():
        print(f"  {cat}: {diff:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate harassment and misogyny detection models")
    parser.add_argument('--mode', choices=['full', 'edge', 'compare', 'all'], 
                       default='all', help='Evaluation mode')
    
    args = parser.parse_args()
    
    try:
        if args.mode in ['full', 'all']:
            evaluate_models_on_datasets()
        
        if args.mode in ['edge', 'all']:
            test_edge_cases()
        
        if args.mode in ['compare', 'all']:
            compare_models_performance()
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()