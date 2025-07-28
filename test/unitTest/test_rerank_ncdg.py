import unittest
import pandas as pd
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Path setup
current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Debug imports
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

try:
    from re_rank.core import Reranker  # Assumes Reranker has __call__(query, passages) method
    print("✓ Reranker imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise

class RerankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load cleaned CSV
        csv_path = os.path.join(current_path, "rerank_flat_benchmark.csv")
        cls.df = pd.read_csv(csv_path)
        print(f"Loaded {len(cls.df)} rows from {csv_path}")
        
        # Verify data structure
        unique_queries = cls.df['query_id'].nunique()
        print(f"Found {unique_queries} unique queries")
        
        # Check rank distribution
        rank_counts = cls.df['rank'].value_counts().sort_index()
        print(f"Rank distribution: {dict(rank_counts)}")
        
        cls.reranker = Reranker()
        print("✓ Reranker initialized")

    def dcg(self, rels):
        """Calculate Discounted Cumulative Gain"""
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rels))

    def ndcg(self, true_ranks, predicted_scores):
        """Calculate Normalized Discounted Cumulative Gain"""
        # Convert rank to relevance score: 1 → 3, 2 → 2, 3 → 1
        rel_map = {1: 3, 2: 2, 3: 1}
        relevance = [rel_map.get(rank, 0) for rank in true_ranks]

        # Predicted ranking indices (sorted by score, highest first)
        predicted_order = sorted(range(len(predicted_scores)), key=lambda i: predicted_scores[i], reverse=True)
        pred_rels = [relevance[i] for i in predicted_order]

        # Ideal ranking (sorted by highest relevance)
        ideal_rels = sorted(relevance, reverse=True)

        dcg_val = self.dcg(pred_rels)
        idcg_val = self.dcg(ideal_rels)
        return dcg_val / idcg_val if idcg_val > 0 else 0.0

    def test_rank_and_ndcg(self):
        """Test reranker performance using exact match accuracy and nDCG"""
        grouped = self.df.groupby("query_id")
        total = 0
        exact_matches = 0
        ndcg_scores = []

        for query_id, group in grouped:
            query = group["query"].iloc[0]
            # Shuffle passages to test reranker's ability to recover correct order
            shuffled = group.sample(frac=1, random_state=42).reset_index(drop=True)
            passages = shuffled["passage"].tolist()
            true_ranks = shuffled["rank"].tolist()

            try:
                # Use __call__ method which returns (ranked_scores, ranked_passages)
                ranked_scores, ranked_passages = self.reranker(query, passages)
                
                # Validate results
                self.assertEqual(len(ranked_scores), len(passages), 
                               f"Reranker returned {len(ranked_scores)} scores for {len(passages)} passages")
                self.assertEqual(len(ranked_passages), len(passages), 
                               f"Reranker returned {len(ranked_passages)} passages for {len(passages)} input passages")
                self.assertTrue(all(isinstance(s, (int, float)) for s in ranked_scores), 
                               "All scores must be numeric")
                
                # Create mapping from original passages to their predicted ranks
                passage_to_predicted_rank = {}
                for rank, passage in enumerate(ranked_passages, 1):
                    passage_to_predicted_rank[passage] = rank
                
                # Get predicted ranks in the same order as input passages
                predicted_ranks = [passage_to_predicted_rank[passage] for passage in passages]
                
                # For nDCG calculation, we need the original scores in input order
                # Create mapping from passages to their scores
                passage_to_score = dict(zip(ranked_passages, ranked_scores))
                original_order_scores = [passage_to_score[passage] for passage in passages]
                
            except Exception as e:
                self.fail(f"Reranker failed on query_id {query_id} with error: {str(e)}")

            # Check exact match (comparing rank arrays)
            is_exact = predicted_ranks == true_ranks
            
            if is_exact:
                exact_matches += 1
            
            # Calculate nDCG using original order scores
            ndcg_val = self.ndcg(true_ranks, original_order_scores)
            ndcg_scores.append(ndcg_val)

            # Print detailed results
            print(f"[{query_id[:8]}] Exact Match: {'✔' if is_exact else '✘'} | nDCG: {ndcg_val:.4f}")
            print(f"  Query: {query[:60]}{'...' if len(query) > 60 else ''}")
            print(f"  True ranks:  {true_ranks}")
            print(f"  Pred ranks:  {predicted_ranks}")
            if not is_exact:
                print(f"  Scores:      {[f'{s:.3f}' for s in original_order_scores]}")
                print(f"  Ranked order: {[passages.index(p) + 1 for p in ranked_passages]}")
            print()
            
            total += 1

        # Calculate final metrics
        exact_acc = exact_matches / total if total else 0
        avg_ndcg = sum(ndcg_scores) / total if total else 0

        print("=" * 60)
        print(f"FINAL RESULTS:")
        print(f"Exact Match Accuracy: {exact_acc:.2%} ({exact_matches}/{total})")
        print(f"Average nDCG: {avg_ndcg:.4f}")
        print("=" * 60)

        # Assertions with informative messages
        self.assertGreater(avg_ndcg, 0.7, 
                          f"nDCG too low: {avg_ndcg:.4f} <= 0.7. Reranker performance is insufficient.")
        self.assertGreater(exact_acc, 0.3, 
                          f"Too few exact matches: {exact_acc:.2%} <= 30%. Only {exact_matches}/{total} queries had perfect ranking.")

    def test_individual_query_debugging(self):
        """Helper test for debugging individual queries"""
        # Enable this for detailed debugging of specific queries
        DEBUG_MODE = False
        
        if not DEBUG_MODE:
            self.skipTest("Debug mode disabled")
            
        grouped = self.df.groupby("query_id")
        
        for query_id, group in grouped:
            query = group["query"].iloc[0]
            passages = group["passage"].tolist()
            true_ranks = group["rank"].tolist()
            
            print(f"\n{'='*80}")
            print(f"DEBUG: Query ID {query_id}")
            print(f"Query: {query}")
            print(f"True ranks: {true_ranks}")
            
            for i, (passage, rank) in enumerate(zip(passages, true_ranks)):
                print(f"\nPassage {i+1} (Rank {rank}):")
                print(f"  {passage[:200]}{'...' if len(passage) > 200 else ''}")
            
            try:
                ranked_scores, ranked_passages = self.reranker(query, passages)
                
                print(f"\nRanked Results:")
                for i, (score, passage) in enumerate(zip(ranked_scores, ranked_passages)):
                    original_idx = passages.index(passage) + 1
                    original_rank = true_ranks[passages.index(passage)]
                    print(f"  {i+1}. Passage {original_idx} (true rank {original_rank}) - Score: {score:.4f}")
                    print(f"     {passage[:100]}{'...' if len(passage) > 100 else ''}")
                    
            except Exception as e:
                print(f"ERROR: {e}")
            
            break  # Only debug first query

if __name__ == "__main__":
    unittest.main()