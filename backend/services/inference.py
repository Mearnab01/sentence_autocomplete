import torch
import time
from models.dl_model import LSTMModel
from models.dataset import TextSequenceDataset
from models.llm_model import get_llm_completion
from utils.cache import global_cache
from utils.logger import get_logger

logger = get_logger(__name__)

# Basic global state for DL model (in a real app, use dependency injection or app state)
dl_model = None
dl_dataset = None  # To access vocab

def load_dl_model(vocab_size: int, model_path: str = "lstm_model.pth"):
    global dl_model
    dl_model = LSTMModel(vocab_size=vocab_size)
    try:
        if torch.cuda.is_available():
            dl_model.load_state_dict(torch.load(model_path))
        else:
            dl_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        dl_model.eval()
        logger.info(f"Loaded DL model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load DL model from {model_path}, using untrained weights: {e}")

async def predict_llm(prompt: str) -> dict:
    start_time = time.time()
    
    # Check cache
    cache_key = f"llm_{prompt}"
    cached_result = global_cache.get(cache_key)
    if cached_result:
        return cached_result
        
    completion = await get_llm_completion(prompt)
    latency = time.time() - start_time
    
    result = {
        "suggestions": [completion] if completion else [],
        "latency_ms": round(latency * 1000, 2),
        "mode": "llm_mode",
        "source": "api"
    }
    
    if completion:
        global_cache.set(cache_key, result)
        
    return result

def predict_dl(prompt: str, top_k: int = 3) -> dict:
    start_time = time.time()
    
    # Check cache
    cache_key = f"dl_{prompt}_{top_k}"
    cached_result = global_cache.get(cache_key)
    if cached_result:
        return cached_result

    if dl_model is None or dl_dataset is None:
        logger.error("DL model or vocabulary not initialized")
        return {
            "suggestions": [],
            "latency_ms": 0,
            "mode": "dl_mode",
            "error": "Model not initialized. Please run training first."
        }

    # Tokenize prompt using same vocab
    words = prompt.lower().split()
    # Take last seq_length words, or pad if too short
    seq_length = 5
    tokens = [dl_dataset.vocab.get(w, dl_dataset.vocab["<unk>"]) for w in words[-seq_length:]]
    
    # Pad if necessary
    while len(tokens) < seq_length:
        tokens.insert(0, dl_dataset.vocab["<pad>"])
        
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    
    dl_model.eval()
    with torch.no_grad():
        output, _ = dl_model(input_tensor)
        # Get top-k probabilities
        probs = torch.softmax(output, dim=-1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        suggestions = []
        for idx in top_indices:
            word = dl_dataset.idx2word.get(idx.item(), "")
            if word and word not in ["<pad>", "<unk>"]:
                suggestions.append(word)
                
    latency = time.time() - start_time
    result = {
        "suggestions": suggestions,
        "latency_ms": round(latency * 1000, 2),
        "mode": "dl_mode",
        "source": "local_model"
    }
    
    global_cache.set(cache_key, result)
    return result
