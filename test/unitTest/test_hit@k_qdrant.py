import unittest
import pandas as pd
import time
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai


current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

from rag.core import RAG



class HitAtKTest(unittest.TestCase):

    output_dir = os.path.join(project_root, 'test', 'unitTest', 'eval_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    @classmethod
    def setUpClass(cls):
        """Set up the environment, RAG object, and load test data."""
        
        dotenv_path = os.path.join(project_root, '.env')
        load_dotenv(dotenv_path=dotenv_path)
        
        llm_key = os.getenv('GEMINI_KEY')
        QDRANT_API = os.getenv('QDRANT_API')
        QDRANT_URL = os.getenv('QDRANT_URL')


        if not all([llm_key]):
            raise ValueError("One or more required environment variables are missing in the .env file.")

        genai.configure(api_key=llm_key)
        llm = genai.GenerativeModel('gemini-1.5-pro')
        
        cls.rag_instance = RAG(
            type='qdrant',
            qdrant_api=QDRANT_API,
            qdrant_url=QDRANT_URL,
            embeddingName='Alibaba-NLP/gte-multilingual-base',
            llm=llm,
        )
        
        try:
            csv_path = os.path.join(current_path, "hit_k_benchmark.csv")
            cls.df = pd.read_csv(csv_path)
            print(f"Loaded {len(cls.df)} test cases from {csv_path}")
        except FileNotFoundError:
            raise unittest.SkipTest(f"hit_k_benchmark.csv not found at {csv_path}")

    def save_result_to_csv(self, result: dict, filename: str):
        """Save test results to a CSV file."""
        try:
            output_path = os.path.join(HitAtKTest.output_dir, filename)
            file_exists = os.path.isfile(output_path)
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"Warning: Failed to save result to CSV: {e}")

    def test_hit_at_1(self):
        """
        Execute vector search for all queries and calculate Hit@1.
        """
        results_filename = "log_hit_at_1_test_results.csv"
        log_file_path = os.path.join(HitAtKTest.output_dir, results_filename)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        
        hits = []
        
        print(f"\nStarting Hit@1 evaluation for {len(self.df)} test cases...")
        print("=" * 80)
        
        for index, row in self.df.iterrows():
            with self.subTest(row_index=index):
                query = str(row.get('query', '')).strip()
                ground_truth_id = str(row.get('ground_truth_id', '')).strip()
                
                if not query or not ground_truth_id:
                    print(f"Row {index}: Skipping - missing query or ground_truth_id")
                    continue
                
                print(f"Row {index}: Testing query: '{query[:60]}...'")
                
                top1_search_id = None
                score = 0
                search_time = 0
                
                try:
                    start_time = time.time()
                    search_results = self.rag_instance.vector_search(query, limit=1)
                    
                    print('--->search_results', search_results)
                    
                    search_time = time.time() - start_time
                    
                    if search_results:
                        # Extract ID from search results based on the backend type
                        if hasattr(self.rag_instance, 'type') and self.rag_instance.type == 'qdrant':
                            # For Qdrant, the ID should be in the payload
                            top1_search_id = str(search_results[0].get('_id', search_results[0].get('id', 'ID_NOT_FOUND')))
                        else:
                            # For MongoDB, the ID is directly accessible
                            top1_search_id = str(search_results[0].get('_id', 'ID_NOT_FOUND'))
                        
                        if top1_search_id == ground_truth_id:
                            score = 1
                    
                    hits.append(score)

                    print(f"Row {index}: Ground Truth ID: '{ground_truth_id}'")
                    print(f"Row {index}: Top 1 Search ID: '{top1_search_id}'")
                    print(f"Row {index}: Score = {score} | Search Time: {search_time:.2f}s")

                except Exception as e:
                    print(f"Row {index}: Error during search - {str(e)}")
                    hits.append(0) 
                
                finally:
                    result_data = {
                        "row_index": index,
                        "query": query,
                        "ground_truth_id": ground_truth_id,
                        "top1_search_id": top1_search_id or "N/A",
                        "score": score,
                        "search_time_second": round(search_time, 4),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.save_result_to_csv(result_data, results_filename)
                    print("-" * 80)
                    time.sleep(0.5)

        if hits:
            total_tests = len(hits)
            total_hits = sum(hits)
            hit_rate = total_hits / total_tests if total_tests > 0 else 0
            
            print("\n" + "=" * 80)
            print("HIT@1 RETRIEVAL SUMMARY:")
            print(f"Total test cases executed: {total_tests}")
            print(f"Total hits (Top 1 matches Ground Truth): {total_hits}")
            print(f"Hit@1 Rate: {hit_rate:.2%} ({total_hits}/{total_tests})")
            print(f"\nDetailed results saved to: {log_file_path}")
            print("=" * 80)
        else:
            self.fail("No successful tests were completed.")

if __name__ == '__main__':
    unittest.main(verbosity=2)