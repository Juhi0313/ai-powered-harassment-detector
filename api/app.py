import sys
sys.path.append('..')

from flask import Flask, jsonify
from flask_cors import CORS
import logging

from config.config import API_CONFIG
from api.routes import api, initialize_predictor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    
    CORS(app)
    
    
    app.register_blueprint(api, url_prefix='/api')
    
    
    @app.route('/')
    def index():
        return jsonify({
            'name': 'AI-Powered Harassment & Misogyny Detector API',
            'version': '1.0.0',
            'description': 'Detect harassment and misogyny in social media comments using RoBERTa models',
            'endpoints': {
                'health': 'GET /api/health',
                'models_info': 'GET /api/models/info',
                'analyze_single': 'POST /api/analyze',
                'analyze_batch': 'POST /api/analyze/batch',
                'filter_toxic': 'POST /api/analyze/filter'
            },
            'documentation': 'See README.md for detailed API documentation'
        }), 200
    
    return app


def main():
    """Main function to run the API server."""
    print("="*70)
    print("AI-POWERED HARASSMENT & MISOGYNY DETECTOR API")
    print("="*70)
    
    
    print("\nInitializing models...")
    if not initialize_predictor():
        print("\nERROR: Failed to initialize models!")
        print("\nPlease ensure you have trained both models:")
        print("  1. python training/train_harassment_model.py")
        print("  2. python training/train_misogyny_model.py")
        print("\nOr check that the model files exist in:")
        print("  - models/harassment_model/")
        print("  - models/misogyny_model/")
        return
    
    print("Models initialized successfully!")
    
    
    app = create_app()
    
    
    print("\n" + "="*70)
    print("STARTING API SERVER")
    print("="*70)
    print(f"\nServer running at: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("\nAvailable endpoints:")
    print("  GET  /                      - API information")
    print("  GET  /api/health            - Health check")
    print("  GET  /api/models/info       - Model information")
    print("  POST /api/analyze           - Analyze single comment")
    print("  POST /api/analyze/batch     - Analyze multiple comments")
    print("  POST /api/analyze/filter    - Filter toxic comments")
    print("\nPress CTRL+C to stop the server")
    print("="*70 + "\n")
    
    
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )


if __name__ == '__main__':
    main()