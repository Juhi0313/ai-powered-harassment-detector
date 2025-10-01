from flask import Blueprint, request, jsonify
from typing import Dict, List
import traceback

from api.predictor import get_predictor

api = Blueprint('api', __name__)

predictor = None

def initialize_predictor():
    """Initialize the predictor (called when app starts)."""
    global predictor
    try:
        predictor = get_predictor()
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False


@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None and predictor.models_loaded
    }), 200


@api.route('/models/info', methods=['GET'])
def get_models_info():
    """Get information about the loaded models."""
    try:
        if predictor is None:
            return jsonify({
                'error': 'Models not initialized'
            }), 503
        
        info = predictor.get_models_info()
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api.route('/analyze', methods=['POST'])
def analyze_single():
    """
    Analyze a single comment for harassment and misogyny.
    
    Request body:
    {
        "text": "comment text to analyze"
    }
    
    Response:
    {
        "text": "original comment",
        "harassment_score": 0.85,
        "misogyny_score": 0.72,
        "combined_toxicity_score": 0.794,
        "is_harassment": true,
        "is_misogyny": true,
        "is_toxic": true,
        "risk_level": "high",
        "details": {...}
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Models not initialized. Please ensure models are trained and available.'
            }), 503
        
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text must be a non-empty string'
            }), 400
        
        
        result = predictor.predict_single(text)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple comments for harassment and misogyny.
    
    Request body:
    {
        "texts": ["comment 1", "comment 2", ...],
        "include_statistics": true  // optional, default: false
    }
    
    Response:
    {
        "results": [
            {
                "text": "comment 1",
                "harassment_score": 0.85,
                ...
            },
            ...
        ],
        "statistics": {  // only if include_statistics is true
            "total_comments": 10,
            "toxic_comments": 3,
            ...
        }
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Models not initialized. Please ensure models are trained and available.'
            }), 503
        
       
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts'
            }), 400
        
        texts = data['texts']
        include_statistics = data.get('include_statistics', False)
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts must be a list of strings'
            }), 400
        
        if len(texts) == 0:
            return jsonify({
                'error': 'texts list cannot be empty'
            }), 400
        
        if len(texts) > 100:
            return jsonify({
                'error': 'Maximum 100 texts allowed per batch request'
            }), 400
        
        
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return jsonify({
                    'error': f'Text at index {i} is invalid (must be non-empty string)'
                }), 400
        
       
        results = predictor.predict_batch(texts)
        
        response = {
            'results': results
        }
        
        
        if include_statistics:
            stats = predictor.get_batch_statistics(results)
            response['statistics'] = stats
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api.route('/analyze/filter', methods=['POST'])
def filter_toxic_comments():
    """
    Filter comments based on toxicity threshold.
    
    Request body:
    {
        "texts": ["comment 1", "comment 2", ...],
        "threshold": 0.5,  // optional, default: 0.5
        "filter_type": "all"  // optional: "all", "harassment", "misogyny"
    }
    
    Response:
    {
        "total_comments": 10,
        "toxic_comments": 3,
        "filtered_results": [
            {
                "index": 2,
                "text": "toxic comment",
                "combined_toxicity_score": 0.85,
                ...
            }
        ]
    }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Models not initialized'
            }), 503
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts'
            }), 400
        
        texts = data['texts']
        threshold = data.get('threshold', 0.5)
        filter_type = data.get('filter_type', 'all')
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'error': 'texts must be a non-empty list'
            }), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({
                'error': 'threshold must be between 0 and 1'
            }), 400
        
        if filter_type not in ['all', 'harassment', 'misogyny']:
            return jsonify({
                'error': 'filter_type must be one of: all, harassment, misogyny'
            }), 400
        
        
        results = predictor.predict_batch(texts)
        
        
        filtered_results = []
        for i, result in enumerate(results):
            is_toxic = False
            
            if filter_type == 'all':
                is_toxic = result['combined_toxicity_score'] >= threshold
            elif filter_type == 'harassment':
                is_toxic = result['harassment_score'] >= threshold
            elif filter_type == 'misogyny':
                is_toxic = result['misogyny_score'] >= threshold
            
            if is_toxic:
                result['index'] = i
                filtered_results.append(result)
        
        return jsonify({
            'total_comments': len(texts),
            'toxic_comments': len(filtered_results),
            'threshold': threshold,
            'filter_type': filter_type,
            'filtered_results': filtered_results
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@api.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405


@api.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500