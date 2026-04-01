import os
import aiohttp
from utils.logger import get_logger

logger = get_logger(__name__)

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

async def get_llm_completion(prompt: str, max_new_tokens: int = 5, temperature: float = 0.5) -> str:
    if not HF_API_KEY:
        logger.warning("HUGGINGFACE_API_KEY not set. Using dummy fallback response.")
        return " (dummy LLM suggestion)"

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    formatted_prompt = f"<s>[INST] Continue the following text naturally. Provide ONLY the next few words, without any explanation. Text: '{prompt}' [/INST]"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(HF_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        completion = result[0].get("generated_text", "")
                        return completion.strip()
                else:
                    text = await response.text()
                    logger.error(f"HF API returned {response.status}: {text}")
    except Exception as e:
        logger.error(f"Error calling Hugging Face API: {e}")
        
    return ""
