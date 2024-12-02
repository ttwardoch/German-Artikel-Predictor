import torch
import torch.nn as nn


class ArtikelTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers):
        super(ArtikelTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 100, embedding_dim))  # Assuming max length of 100
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_indices = nn.Linear(embedding_dim, 4)

    def forward(self, x):
        seq_len = x.size(1)
        pos_embedded = self.pos_embedding[:, :seq_len, :]
        embedded = self.embedding(x) + pos_embedded
        transformer_out = self.transformer(embedded)
        pooled_out = transformer_out.mean(dim=1)
        return self.fc_indices(pooled_out)