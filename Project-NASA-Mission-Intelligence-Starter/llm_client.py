from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # 1. Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # 2. Define system prompt and set context
    # We place the context here so the model knows its constraints/knowledge base
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use this context: {context}"}
    ]

    # 3. Add chat history
    # This assumes conversation_history is a list of {'role': '...', 'content': '...'}
    messages.extend(conversation_history)

    # 4. Add the current user message
    messages.append({"role": "user", "content": user_message})

    # 5. Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7  # Optional: adjusts creativity
        )

        # 6. Return response text
        return response.choices[0].message.content
        
    except Exception as e:
        return f"An error occurred: {e}"