import numpy as np
import torch
from tqdm import tqdm

def get_embeddings(texts, model, tokenizer, device, model_type="base"):
    """Extract embeddings from model"""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 8), desc=f"Extracting {model_type} embeddings"):
            batch = texts[i:i+8]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            
            # Get hidden states
            outputs = model(**enc)
            hidden_states = outputs.last_hidden_state
            
            # Use mean pooling for better representation
            embeddings_batch = hidden_states.mean(dim=1)
            embeddings.append(embeddings_batch.cpu().numpy())
    
    return np.vstack(embeddings)

def get_llm_embeddings(texts, model, tokenizer, device, model_type="base", batch_size=4):
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {model_type} embeddings"):
            batch = texts[i:i+batch_size]
            
            # Tokenize batch
            encoded = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(model.device)
            attention_mask = encoded['attention_mask'].to(model.device)
            
            # Get hidden states
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Mean pooling over sequence length (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings_batch = sum_embeddings / sum_mask
            
            embeddings.append(embeddings_batch.cpu().numpy())
    
    return np.vstack(embeddings)