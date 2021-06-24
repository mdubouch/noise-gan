import torch
import torch.nn as nn
import torch.nn.functional as F

class CDCEmbedding(nn.Module):
    def __init__(self, context_size, hidden_size, n_wires):
        super().__init__()
        self.emb = nn.Embedding(n_wires, hidden_size, sparse=False)
        self.lin = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, n_wires)
        )
    def forward(self, x):
        # shape=(batch, context_size)
        # Embed the neighbors
        emb_x = self.emb(x) # shape=(batch, context_size, hidden_size)
        # Compute the average point from the neighbors
        avg_emb = emb_x.mean(dim=1) # shape=(batch, hidden_size)
        # Invert the embedding
        out = self.lin(avg_emb) # shape=(batch, n_wires)
        return out
