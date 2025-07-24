import unittest
import pandas as pd
import requests
import json
import time
import os
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics import RougeScore



class SearchAPIROUGETest(unittest.TestCase):
    """Test suite for Search API - calculates ROUGE scores using RAGAS"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test data and RAGAS ROUGEScore scorer"""
        cls.api_url = "http://localhost:5002/api/search"
        cls.headers = {"Content-Type": "application/json"}
        
        # Initialize RAGAS RougeScore scorer
        cls.rouge_scorer = RougeScore()

        
        # Load CSV data
        try:
            current_path = os.path.dirname(__file__)
            file_path = os.path.join(current_path, "question_answer.csv")
            cls.df = pd.read_csv(file_path)
            print(f"Loaded {len(cls.df)} test cases from CSV")
        except FileNotFoundError:
            raise unittest.SkipTest("question_answer.csv not found")
    
    def call_search_api(self, query: str) -> str:
        """Make API call to the search endpoint"""
        payload = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        ]
        
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract text from response structure
            if "parts" in response_data and len(response_data["parts"]) > 0:
                return response_data["parts"][0].get("text", "")
            else:
                return ""
                
        except requests.exceptions.RequestException as e:
            self.fail(f"API call failed: {e}")
        except json.JSONDecodeError as e:
            self.fail(f"Failed to parse JSON response: {e}")
    
    async def calculate_rouge_score_async(self, reference: str, response: str) -> float:
        """Calculate ROUGE-L F1 score using RAGAS RougeScore"""
        try:
            if not reference or not response:
                return 0.0
            
            # Create SingleTurnSample for RAGAS
            sample = SingleTurnSample(
                response=response.strip(),
                reference=reference.strip()
            )
            
            # Calculate ROUGE score using RAGAS
            rouge_score = await self.rouge_scorer.single_turn_ascore(sample)
            return float(rouge_score)
            
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
            return 0.0
    
    def calculate_rouge_score(self, reference: str, response: str) -> float:
        """Synchronous wrapper for async ROUGE score calculation"""
        try:
            # Create new event loop for this calculation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.calculate_rouge_score_async(reference, response))
            loop.close()
            return result
        except Exception as e:
            print(f"Error in ROUGE score calculation wrapper: {e}")
            return 0.0
    
    def save_result_to_csv(self, result: dict, filename: str = "log_rouge_test_results.csv"):
        """Save test result to CSV file"""
        try:
            file_exists = os.path.isfile(filename)
            result_df = pd.DataFrame([result])
            result_df.to_csv(filename, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"Warning: Failed to save result to CSV: {e}")
    
    def test_rouge_scores_all_queries(self):
        """Test all queries from CSV and calculate ROUGE scores using RAGAS"""
        
        # Initialize results file
        results_filename = "./eval_dataset/log_rouge_test_results.csv"
        if os.path.exists(results_filename):
            os.remove(results_filename)
        
        total_rouge_scores = []
        successful_tests = 0
        
        print(f"\nStarting ROUGE score evaluation using RAGAS for {len(self.df)} test cases...")
        print("=" * 80)
        
        for index, row in self.df.iterrows():
            with self.subTest(row_index=index):
                query = str(row.get('query', '')).strip()
                expected_answer = str(row.get('answer', '')).strip()
                
                if not query or not expected_answer or expected_answer.lower() in ['nan', 'null', 'none']:
                    print(f"Row {index}: Skipping - missing query or answer")
                    continue
                
                print(f"Row {index}: Testing query: '{query[:50]}...'")
                
                try:
                    # Call API
                    start_time = time.time()
                    api_response = self.call_search_api(query)
                    api_call_time = time.time() - start_time
                    
                    # Calculate ROUGE score using RAGAS
                    rouge_start_time = time.time()
                    rouge_score = self.calculate_rouge_score(expected_answer, api_response)
                    rouge_calc_time = time.time() - rouge_start_time
                    
                    total_response_time = api_call_time + rouge_calc_time
                    
                    total_rouge_scores.append(rouge_score)
                    successful_tests += 1
                    
                    # Prepare result data
                    result = {
                        "row_index": index,
                        "query": query,
                        "expected_answer": expected_answer,
                        "api_response": api_response,
                        "rouge_score": round(rouge_score, 6),
                        "api_response_time_seconds": round(api_call_time, 2),
                        "rouge_calc_time_seconds": round(rouge_calc_time, 2),
                        "total_time_seconds": round(total_response_time, 2),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Save result
                    self.save_result_to_csv(result, results_filename)
                    
                    # Print progress
                    print(f"Row {index}: RAGAS ROUGE Score = {rouge_score:.6f}")
                    print(f"API Time: {api_call_time:.2f}s | ROUGE Calc Time: {rouge_calc_time:.2f}s")
                    print(f"Expected: {expected_answer[:80]}...")
                    print(f"Got:      {api_response[:80]}...")
                    print("-" * 80)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Row {index}: Error - {str(e)}")
                    # Save error result
                    error_result = {
                        "row_index": index,
                        "query": query,
                        "expected_answer": expected_answer,
                        "api_response": f"ERROR: {str(e)}",
                        "rouge_score": 0.0,
                        "api_response_time_seconds": 0.0,
                        "rouge_calc_time_seconds": 0.0,
                        "total_time_seconds": 0.0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.save_result_to_csv(error_result, results_filename)
                    continue
        
        # Calculate and print summary statistics
        if total_rouge_scores:
            avg_rouge = sum(total_rouge_scores) / len(total_rouge_scores)
            max_rouge = max(total_rouge_scores)
            min_rouge = min(total_rouge_scores)
            
            print("\n" + "=" * 80)
            print("RAGAS ROUGE SCORE SUMMARY:")
            print(f"Total test cases: {len(self.df)}")
            print(f"Successful tests: {successful_tests}")
            print(f"Average ROUGE Score: {avg_rouge:.6f}")
            print(f"Maximum ROUGE Score: {max_rouge:.6f}")
            print(f"Minimum ROUGE Score: {min_rouge:.6f}")
            
            # Count scores by range (adjusted for RAGAS ROUGE scoring)
            excellent = sum(1 for score in total_rouge_scores if score >= 0.8)
            good = sum(1 for score in total_rouge_scores if 0.6 <= score < 0.8)
            fair = sum(1 for score in total_rouge_scores if 0.4 <= score < 0.6)
            poor = sum(1 for score in total_rouge_scores if score < 0.4)
            
            print(f"\nScore Distribution:")
            print(f"Excellent (≥0.8): {excellent} ({excellent/len(total_rouge_scores)*100:.1f}%)")
            print(f"Good (0.6-0.8):   {good} ({good/len(total_rouge_scores)*100:.1f}%)")
            print(f"Fair (0.4-0.6):   {fair} ({fair/len(total_rouge_scores)*100:.1f}%)")
            print(f"Poor (<0.4):      {poor} ({poor/len(total_rouge_scores)*100:.1f}%)")
            
            # Calculate median and standard deviation
            sorted_scores = sorted(total_rouge_scores)
            median_rouge = sorted_scores[len(sorted_scores)//2]
            std_dev = (sum((x - avg_rouge) ** 2 for x in total_rouge_scores) / len(total_rouge_scores)) ** 0.5
            
            print(f"\nAdditional Statistics:")
            print(f"Median ROUGE Score: {median_rouge:.6f}")
            print(f"Standard Deviation: {std_dev:.6f}")
            
            print(f"\nDetailed results saved to: {results_filename}")
            print("=" * 80)
            
            # Assert that average ROUGE score meets minimum threshold
            self.assertGreater(avg_rouge, 0.1, f"Average ROUGE score {avg_rouge:.6f} is too low")
            
        else:
            self.fail("No successful tests completed")


def run_example_test():
    """Run the example from RAGAS documentation to verify setup"""
    print("Running RAGAS RougeScore example...")
    
    async def test_example():
        sample = SingleTurnSample(
            response="The Eiffel is located in India.",
            reference="The bbbbbb Eiffel is located in Paris."
        )
        
        scorer = RougeScore()
        score = await scorer.single_turn_ascore(sample)
        print(f"Example ROUGE Score: {score}")
        return score
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_example())
        loop.close()
        
        expected_score = 0.7071067811865478
        if abs(result - expected_score) < 0.0001:
            print("✅ RAGAS RougeScore setup verified successfully!")
        else:
            print(f"⚠️  Expected {expected_score}, got {result}")
            
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("Make sure you have installed ragas: pip install ragas")


if __name__ == '__main__':
    # Check required packages
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import RougeScore
        import pandas as pd
        import requests
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install ragas pandas requests")
        exit(1)
    
    # Run example test first
    run_example_test()
    print()
    
    # Run the main test suite
    unittest.main(verbosity=2)