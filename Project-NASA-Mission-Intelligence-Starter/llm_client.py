from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # 1. Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # 2. Define system prompt and set context
    system_prompt = """You are a senior NASA Mission Expert. Your goal is to provide technical, 
accurate, and concise information based on the provided search context.

RULES:
1. CITATION: Always cite the Document number (e.g., [Document 1]) when referencing facts.
2. LIMITS: Only use the provided context. If the answer isn't there, say: 
   "I'm sorry, my current mission records do not contain specific information to answer that."
3. TONE: Professional, scientific, and helpful.
4. If a mission filter was applied, focus strictly on that mission's parameters."""
    # We place the context here so the model knows its constraints/knowledge base
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # 3. Add chat history
    # This assumes conversation_history is a list of {'role': '...', 'content': '...'}
    messages.extend(conversation_history)

    # 4. Add the current user message
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"})

    # 5. Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )

        # 6. Return response text
        return response.choices[0].message.content
        
    except Exception as e:
        return f"An error occurred: {e}"