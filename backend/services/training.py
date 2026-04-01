import torch
import torch.nn as nn
import torch.optim as optim
import math
from models.dl_model import LSTMModel
from models.dataset import TextSequenceDataset
from utils.logger import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)

def train_dl_model(texts: list[str], epochs: int = 5, batch_size: int = 32, save_path: str = "lstm_model.pth"):
    logger.info(f"Starting training on {len(texts)} texts for {epochs} epochs")
    
    dataset = TextSequenceDataset(texts, seq_length=5, min_freq=1)
    if len(dataset) == 0:
        logger.error("Not enough data to create sequences")
        return {"error": "Not enough data for sequence generation.", "status": "failed"}

    # Update global references to be picked up by inference!
    import services.inference as inference
    inference.dl_dataset = dataset
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    
    model = LSTMModel(vocab_size=vocab_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 50 else float('inf')
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
    # Save Model
    torch.save(model.state_dict(), save_path)
    inference.load_dl_model(vocab_size, save_path)
    
    final_avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    final_ppl = math.exp(final_avg_loss) if final_avg_loss < 50 else float('inf')
    
    logger.info(f"Training completed. Final Loss: {final_avg_loss:.4f}")
    return {
        "status": "success",
        "epochs": epochs,
        "vocab_size": vocab_size,
        "final_loss": round(final_avg_loss, 4),
        "perplexity": round(final_ppl, 4),
        "saved_to": save_path
    }
