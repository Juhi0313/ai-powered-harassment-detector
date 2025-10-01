import requests
import json
from typing import Dict, List


API_BASE_URL = "http://localhost:5000"
API_PREFIX = "/api"


class APITester:
    """Test client for the toxicity detection API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """Test the health check endpoint."""
        print("\n" + "="*70)
        print("Testing Health Check Endpoint")
        print("="*70)
        
        url = f"{self.base_url}{API_PREFIX}/health"
        response = self.session.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    
    def test_models_info(self):
        """Test the models info endpoint."""
        print("\n" + "="*70)
        print("Testing Models Info Endpoint")
        print("="*70)
        
        url = f"{self.base_url}{API_PREFIX}/models/info"
        response = self.session.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.json()
    
    def test_analyze_single(self, text: str):
        """Test analyzing a single comment."""
        print("\n" + "="*70)
        print("Testing Single Comment Analysis")
        print("="*70)
        
        url = f"{self.base_url}{API_PREFIX}/analyze"
        payload = {"text": text}
        
        print(f"Comment: {text}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = self.session.post(url, json=payload)
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\nResults:")
            print(f"  Combined Toxicity Score: {result['combined_toxicity_score']:.4f}")
            print(f"  Harassment Score: {result['harassment_score']:.4f}")
            print(f"  Misogyny Score: {result['misogyny_score']:.4f}")
            print(f"  Is Toxic: {result['is_toxic']}")
            print(f"  Is Harassment: {result['is_harassment']}")
            print(f"  Is Misogyny: {result['is_misogyny']}")
            print(f"  Risk Level: {result['risk_level'].upper()}")
        else:
            print(f"Error: {json.dumps(result, indent=2)}")
        
        return result
    
    def test_analyze_batch(self, texts: List[str], include_statistics: bool = True):
        """Test analyzing multiple comments."""
        print("\n" + "="*70)
        print("Testing Batch Comment Analysis")
        print("="*70)
        
        url = f"{self.base_url}{API_PREFIX}/analyze/batch"
        payload = {
            "texts": texts,
            "include_statistics": include_statistics
        }
        
        print(f"Number of comments: {len(texts)}")
        print(f"Include statistics: {include_statistics}")
        
        response = self.session.post(url, json=payload)
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\nProcessed {len(result['results'])} comments")
            
            if include_statistics and 'statistics' in result:
                stats = result['statistics']
                print(f"\nBatch Statistics:")
                print(f"  Total Comments: {stats['total_comments']}")
                print(f"  Toxic Comments: {stats['toxic_comments']} ({stats['toxic_percentage']:.1f}%)")
                print(f"  Harassment: {stats['harassment_count']} ({stats['harassment_percentage']:.1f}%)")
                print(f"  Misogyny: {stats['misogyny_count']} ({stats['misogyny_percentage']:.1f}%)")
                print(f"\n  Average Scores:")
                print(f"    Harassment: {stats['average_scores']['harassment']:.4f}")
                print(f"    Misogyny: {stats['average_scores']['misogyny']:.4f}")
                print(f"    Combined: {stats['average_scores']['combined']:.4f}")
                print(f"\n  Risk Distribution:")
                for level, count in stats['risk_distribution'].items():
                    print(f"    {level.capitalize()}: {count}")
            
            
            print(f"\nSample Results (first 3):")
            for i, res in enumerate(result['results'][:3]):
                print(f"\n  Comment {i+1}: {res['text'][:60]}...")
                print(f"    Toxicity: {res['combined_toxicity_score']:.4f} | Risk: {res['risk_level']}")
        else:
            print(f"Error: {json.dumps(result, indent=2)}")
        
        return result
    
    def test_filter_toxic(self, texts: List[str], threshold: float = 0.5, filter_type: str = "all"):
        """Test filtering toxic comments."""
        print("\n" + "="*70)
        print("Testing Toxic Comment Filtering")
        print("="*70)
        
        url = f"{self.base_url}{API_PREFIX}/analyze/filter"
        payload = {
            "texts": texts,
            "threshold": threshold,
            "filter_type": filter_type
        }
        
        print(f"Number of comments: {len(texts)}")
        print(f"Threshold: {threshold}")
        print(f"Filter type: {filter_type}")
        
        response = self.session.post(url, json=payload)
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\nFiltering Results:")
            print(f"  Total Comments: {result['total_comments']}")
            print(f"  Toxic Comments: {result['toxic_comments']}")
            
            if result['toxic_comments'] > 0:
                print(f"\n  Filtered Comments:")
                for item in result['filtered_results']:
                    print(f"\n    Index {item['index']}: {item['text'][:60]}...")
                    print(f"    Combined Score: {item['combined_toxicity_score']:.4f}")
                    print(f"    Risk Level: {item['risk_level']}")
        else:
            print(f"Error: {json.dumps(result, indent=2)}")
        
        return result


def run_comprehensive_tests():
    """Run comprehensive API tests."""
    print("\n" + "="*70)
    print("AI-POWERED HARASSMENT & MISOGYNY DETECTOR - API TESTS")
    print("="*70)
    print("\nMake sure the API server is running at http://localhost:5000")
    print("Run: python api/app.py")
    
    
    tester = APITester()
    
    
    test_comments = [
        "You're doing amazing work! Keep it up!",
        "Women shouldn't be in leadership positions",
        "Shut up you stupid woman",
        "I respectfully disagree with your analysis",
        "Get back to the kitchen where you belong",
        "Great contribution to the discussion",
        "You're too emotional to make rational decisions",
        "This is harassment and you know it",
        "Thanks for sharing your perspective",
        "Women are inferior and can't handle this job"
    ]
    
    try:
        # Health check
        tester.test_health()
        
        # Models info
        tester.test_models_info()
        
        #  Single comment analysis
        tester.test_analyze_single("Women are too emotional to be leaders")
        
        #  Batch analysis
        tester.test_analyze_batch(test_comments, include_statistics=True)
        
        # Filter toxic comments
        tester.test_filter_toxic(test_comments, threshold=0.6, filter_type="all")
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server!")
        print("Please ensure the server is running: python api/app.py")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_tests()