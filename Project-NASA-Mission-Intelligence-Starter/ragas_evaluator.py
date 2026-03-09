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
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # 1. Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # 2. Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    # 3. Define an instance for each metric to evaluate
    # Note: NonLLMContextPrecisionWithReference usually requires a reference answer; 
    # if you don't have one, stick to Faithfulness and Relevancy.
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        BleuScore(),
        RougeScore()
    ]
    
    # 4. Create a SingleTurnSample for the individual evaluation
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    
    # 5. Evaluate the response using the metrics
    try:
        # Wrap sample in a dataset as evaluate() expects a collection
        dataset = EvaluationDataset(samples=[sample])
        results = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        # 6. Return the evaluation results as a dictionary
        return results.to_pandas().iloc[0].to_dict()
        
    except Exception as e:
        return {"error": str(e)}
