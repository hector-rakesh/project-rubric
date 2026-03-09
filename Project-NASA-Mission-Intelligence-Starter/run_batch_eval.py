import json
import pandas as pd
import os
from rag_client import initialize_rag_system, retrieve_documents, format_context
from llm_client import generate_response
from ragas_evaluator import evaluate_response_quality
from tqdm import tqdm

def run_batch_evaluation(dataset_path="evaluation_set.json", output_path="ragas_report.csv"):
    # 1. Load Dataset
    with open(dataset_path, "r") as f:
        test_data = json.load(f)
    
    # 2. Setup RAG Backend (Using the first available one as default)
    # Adjust these strings to match your specific Chroma directory/collection
    CHROMA_DIR = "./nasa_embeddings" 
    COLLECTION_NAME = "mission_docs"
    
    collection, success, err = initialize_rag_system(CHROMA_DIR, COLLECTION_NAME)
    if not success:
        print(f"Failed to init: {err}")
        return

    results = []
    api_key = os.getenv("OPENAI_API_KEY")

    print(f"🚀 Starting Evaluation on {len(test_data)} questions...")

    for item in tqdm(test_data):
        question = item["question"]
        
        # Retrieval
        docs_result = retrieve_documents(collection, question, n_results=3)
        contexts = docs_result["documents"][0] if docs_result else []
        formatted_context = format_context(contexts, docs_result["metadatas"][0])
        
        # Generation
        answer = generate_response(api_key, question, formatted_context, [])
        
        # Evaluation
        scores = evaluate_response_quality(question, answer, contexts)
        
        # Record results
        results.append({
            "Mission": item.get("mission", "Unknown"),
            "Question": question,
            "Answer": answer,
            "Faithfulness": scores.get("faithfulness", 0),
            "Relevancy": scores.get("answer_relevancy", 0),
            "Context_Precision": scores.get("context_precision", 0)
        })

    # 3. Save and Report
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Evaluation Complete! Report saved to {output_path}")
    print("\nSummary Statistics:")
    print(df[["Faithfulness", "Relevancy"]].mean())

if __name__ == "__main__":
    run_batch_evaluation()