import sys
sys.path.append('..')

from typing import List, Dict
import numpy as np
from pathlib import Path

from config.config import (
    HARASSMENT_MODEL_DIR,
    MISOGYNY_MODEL_DIR,
    TOXICITY_WEIGHTS,
    HARASSMENT_MODEL_CONFIG,
    MISOGYNY_MODEL_CONFIG
)
from models.base_model import RoBERTaToxicityClassifier


class ToxicityPredictor:
    """
    Combined predictor for harassment and misogyny detection.
    Provides toxicity scores and predictions using both models.
    """
    
    def __init__(self):
        """Initialize both harassment and misogyny models."""
        self.harassment_model = None
        self.misogyny_model = None
        self.models_loaded = False
        
    def load_models(self):
        """Load both trained models."""
        print("Loading harassment detection model...")
        
        if not HARASSMENT_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Harassment model not found at {HARASSMENT_MODEL_DIR}. "
                "Please train the model first using: python training/train_harassment_model.py"
            )
        
        self.harassment_model = RoBERTaToxicityClassifier(
            model_name=HARASSMENT_MODEL_CONFIG['base_model'],
            max_length=HARASSMENT_MODEL_CONFIG['max_length']
        )
        self.harassment_model.load_model(str(HARASSMENT_MODEL_DIR))
        
        print("Loading misogyny detection model...")
        
        if not MISOGYNY_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Misogyny model not found at {MISOGYNY_MODEL_DIR}. "
                "Please train the model first using: python training/train_misogyny_model.py"
            )
        
        self.misogyny_model = RoBERTaToxicityClassifier(
            model_name=MISOGYNY_MODEL_CONFIG['base_model'],
            max_length=MISOGYNY_MODEL_CONFIG['max_length']
        )
        self.misogyny_model.load_model(str(MISOGYNY_MODEL_DIR))
        
        self.models_loaded = True
        print("Both models loaded successfully!")
    
    def predict_single(self, text: str) -> Dict:
        """
        Analyze a single comment for harassment and misogyny.
        
        Args:
            text: Comment text to analyze
            
        Returns:
            Dictionary containing:
            - harassment_score: Toxicity score for harassment (0-1)
            - misogyny_score: Toxicity score for misogyny (0-1)
            - combined_score: Weighted combined toxicity score (0-1)
            - is_harassment: Binary prediction for harassment
            - is_misogyny: Binary prediction for misogyny
            - is_toxic: Overall toxicity classification
            - risk_level: Risk level (low/medium/high/critical)
            - details: Detailed breakdown of predictions
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        
        harassment_result = self.harassment_model.predict_single(text)
        harassment_score = harassment_result['toxicity_score']
        is_harassment = harassment_result['prediction'] == 1
        
        
        misogyny_result = self.misogyny_model.predict_single(text)
        misogyny_score = misogyny_result['toxicity_score']
        is_misogyny = misogyny_result['prediction'] == 1
        
        
        combined_score = (
            TOXICITY_WEIGHTS['harassment_weight'] * harassment_score +
            TOXICITY_WEIGHTS['misogyny_weight'] * misogyny_score
        )
        
        
        is_toxic = is_harassment or is_misogyny
        
        
        risk_level = self._calculate_risk_level(combined_score)
        
        return {
            'text': text,
            'harassment_score': float(harassment_score),
            'misogyny_score': float(misogyny_score),
            'combined_toxicity_score': float(combined_score),
            'is_harassment': bool(is_harassment),
            'is_misogyny': bool(is_misogyny),
            'is_toxic': bool(is_toxic),
            'risk_level': risk_level,
            'details': {
                'harassment': {
                    'score': float(harassment_score),
                    'prediction': bool(is_harassment),
                    'confidence': float(max(harassment_result['probability']))
                },
                'misogyny': {
                    'score': float(misogyny_score),
                    'prediction': bool(is_misogyny),
                    'confidence': float(max(misogyny_result['probability']))
                }
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple comments.
        
        Args:
            texts: List of comment texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results
    
    def _calculate_risk_level(self, score: float) -> str:
        """
        Calculate risk level based on combined toxicity score.
        
        Args:
            score: Combined toxicity score (0-1)
            
        Returns:
            Risk level: 'low', 'medium', 'high', or 'critical'
        """
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def get_batch_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistics for a batch of predictions.
        
        Args:
            results: List of prediction results
            
        Returns:
            Statistics dictionary
        """
        total = len(results)
        harassment_count = sum(1 for r in results if r['is_harassment'])
        misogyny_count = sum(1 for r in results if r['is_misogyny'])
        toxic_count = sum(1 for r in results if r['is_toxic'])
        
        avg_harassment_score = np.mean([r['harassment_score'] for r in results])
        avg_misogyny_score = np.mean([r['misogyny_score'] for r in results])
        avg_combined_score = np.mean([r['combined_toxicity_score'] for r in results])
        
        risk_distribution = {
            'low': sum(1 for r in results if r['risk_level'] == 'low'),
            'medium': sum(1 for r in results if r['risk_level'] == 'medium'),
            'high': sum(1 for r in results if r['risk_level'] == 'high'),
            'critical': sum(1 for r in results if r['risk_level'] == 'critical')
        }
        
        return {
            'total_comments': total,
            'toxic_comments': toxic_count,
            'toxic_percentage': (toxic_count / total * 100) if total > 0 else 0,
            'harassment_count': harassment_count,
            'harassment_percentage': (harassment_count / total * 100) if total > 0 else 0,
            'misogyny_count': misogyny_count,
            'misogyny_percentage': (misogyny_count / total * 100) if total > 0 else 0,
            'average_scores': {
                'harassment': float(avg_harassment_score),
                'misogyny': float(avg_misogyny_score),
                'combined': float(avg_combined_score)
            },
            'risk_distribution': risk_distribution
        }
    
    def get_models_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            'models_loaded': self.models_loaded,
            'harassment_model': None,
            'misogyny_model': None,
            'weights': TOXICITY_WEIGHTS
        }
        
        if self.models_loaded:
            info['harassment_model'] = self.harassment_model.get_model_info()
            info['misogyny_model'] = self.misogyny_model.get_model_info()
        
        return info



_predictor_instance = None

def get_predictor() -> ToxicityPredictor:
    """Get or create the global predictor instance."""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = ToxicityPredictor()
        _predictor_instance.load_models()
    
    return _predictor_instance


if __name__ == "__main__":
    
    print("Testing Toxicity Predictor...")
    
    predictor = ToxicityPredictor()
    predictor.load_models()
    
    test_comments = [
        "You're doing great work!",
        "Women are too emotional to lead",
        "Shut up you stupid woman",
        "I disagree with your point, but I respect your opinion",
        "Get back to the kitchen where you belong"
    ]
    
    print("\n" + "="*70)
    print("ANALYZING COMMENTS")
    print("="*70)
    
    results = predictor.predict_batch(test_comments)
    
    for result in results:
        print(f"\nComment: {result['text']}")
        print(f"Combined Toxicity Score: {result['combined_toxicity_score']:.4f}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Is Harassment: {result['is_harassment']}")
        print(f"Is Misogyny: {result['is_misogyny']}")
        print(f"Harassment Score: {result['harassment_score']:.4f}")
        print(f"Misogyny Score: {result['misogyny_score']:.4f}")
        print("-" * 70)
    
    #  batch stats
    stats = predictor.get_batch_statistics(results)
    print("\n" + "="*70)
    print("BATCH STATISTICS")
    print("="*70)
    print(f"Total Comments: {stats['total_comments']}")
    print(f"Toxic Comments: {stats['toxic_comments']} ({stats['toxic_percentage']:.1f}%)")
    print(f"Harassment: {stats['harassment_count']} ({stats['harassment_percentage']:.1f}%)")
    print(f"Misogyny: {stats['misogyny_count']} ({stats['misogyny_percentage']:.1f}%)")
    print(f"\nAverage Scores:")
    print(f"  Harassment: {stats['average_scores']['harassment']:.4f}")
    print(f"  Misogyny: {stats['average_scores']['misogyny']:.4f}")
    print(f"  Combined: {stats['average_scores']['combined']:.4f}")