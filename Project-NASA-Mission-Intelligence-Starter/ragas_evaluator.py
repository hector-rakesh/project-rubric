import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics with strengthened error handling"""
    
    # 1. FIX: Strengthened input validation (The "Guard Clause")
    # Returns clear structured errors instead of failing mid-execution
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    if not answer or not answer.strip():
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "error": "Empty assistant response"}
        
    if not contexts or len(contexts) == 0:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "error": "No context retrieved"}

    try:
        # 2. Setup Evaluators
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
        
        # 3. Define Metrics
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            BleuScore(),
            RougeScore()
        ]
        
        # 4. FIX: Align SingleTurnSample fields
        # Note: In latest Ragas, the field is often 'retrieval_context' 
        # but check your specific version. Usually 'retrieved_contexts' or 'contexts'.
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts  
        )
        
        # 5. Execute Evaluation
        dataset = EvaluationDataset(samples=[sample])
        results = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        # 6. Return Structured Output
        # We convert to dict and remove any internal Ragas metadata for a clean UI/CSV output
        eval_dict = results.to_pandas().iloc[0].to_dict()
        
        # Clean up the dict to ensure only metrics and errors are returned
        return {k: v for k, v in eval_dict.items() if not k.startswith('_')}
        
    except Exception as e:
        # 7. FIX: Ensure malformed inputs during Ragas run return structured data
        return {
            "faithfulness": 0.0, 
            "answer_relevancy": 0.0, 
            "error": f"Evaluation failed: {str(e)}"
        }