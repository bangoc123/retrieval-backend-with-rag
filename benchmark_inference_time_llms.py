import time
import pandas as pd
from llms.llms import LLMs 
import argparse
import os 

class URLNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class APINotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class ValueNotFoundError(Exception):
    def __init__(self, name):
        self.name = name 
        super().__init__(f"Please make sure you have {name} in .env")

class DataNotFoundError(Exception):
    def __init__(self):
        super().__init__(f"Please make sure you have valid CSV file in folder data")

def map_environment_variables(args):
    if args.mode == "online" and args.model_name == "gemini":
        MODEL_API_KEY = os.getenv('GEMINI_API_KEY', None)  
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('GEMINI_API_KEY')

    elif args.mode == "online" and args.model_name == "openai":
        MODEL_API_KEY = os.getenv('OPENAI_API_KEY')
        MODEL_BASE_URL = None

        if not MODEL_API_KEY:
            raise APINotFoundError('OPENAI_API_KEY')

    elif args.mode == "online" and args.model_name == "together":
        MODEL_API_KEY = os.getenv('TOGETHER_API_KEY', None)
        MODEL_BASE_URL = os.getenv("TOGETHER_BASE_URL", None)

        if not MODEL_API_KEY:
            raise APINotFoundError('TOGETHER_API_KEY')
        if not MODEL_BASE_URL:
            raise URLNotFoundError('TOGETHER_BASE_URL')

    elif args.mode == "offline" and args.model_engine == "ollama":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("OLLAMA_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("OLLAMA_BASE_URL")
    
    elif args.mode == "offline" and args.model_engine == "vllm":
        MODEL_API_KEY = None
        MODEL_BASE_URL = os.getenv("VLLM_BASE_URL", None)

        if not MODEL_BASE_URL:
            raise URLNotFoundError("VLLM_BASE_URL")

    elif args.mode == "offline" and args.model_engine == "huggingface":
        MODEL_API_KEY = None
        MODEL_BASE_URL = None
    
    return MODEL_API_KEY, MODEL_BASE_URL

def benchmark_from_csv(args):

    print("Load benchmark data from csv file.\n")
    try:
        df = pd.read_csv(args.csv_path)
    except:
        raise DataNotFoundError

    MODEL_API_KEY, MODEL_BASE_URL = map_environment_variables(args)

    results = []

    total_time = 0

    # Initialize llm
    print("Initialize llm")
    llm = LLMs(type=args.mode, model_version=args.model_version, model_name=args.model_name, engine=args.model_engine, api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)

    for idx, row in df.iterrows():
        data = {"role": "user", "content": row["query"]}
        query_id = row["_id"]   

        print(f"\n[Query ID: {query_id}] Running inference...")

        start_time = time.time()
        output = llm.generate_content([data])
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        print(f"Inference time: {elapsed_time:.4f} seconds")

        results.append({
            "id": query_id,
            "query": row["query"],
            "response": output,
            "inference_time": elapsed_time
        })

    # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"benchmark_results_{args.model_version.split('/')[-1]}_{args.model_engine}.csv", index=False)
    print(f"\n✅ Benchmarking completed. Results saved to benchmark_inference_time_{args.model_version.split('/')[-1]}_{args.model_engine}.csv.")
    print(f"Total time to run inference for {len(df)} queries in dataset: {total_time}")
    print(f"Average time to run inference for {len(df)} queries in dataset: {(total_time/len(df)):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for serve.py")

    parser.add_argument('-c','--csv_path', type=str, required=True, help='Paste your csv file absolute path.')
    parser.add_argument('-m','--mode', type=str, choices=['online', 'offline'], default='offline', help='Choose either online or offline mode system')
    parser.add_argument('-n','--model_name', type=str, default=None, help='Define name of LLM model to use')
    parser.add_argument('-e','--model_engine', type=str, default='huggingface', help='Define model engine of LLM model (Optional)')
    parser.add_argument('-v','--model_version', type=str, required=True, help='Define model version of LLM model (Optional)')

    args = parser.parse_args()
    benchmark_from_csv(args)