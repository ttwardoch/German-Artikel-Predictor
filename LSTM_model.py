import torch
import torch.nn as nn


class ArtikelLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, num_layers):
        super(ArtikelLSTM, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout, bidirectional=True,
                            num_layers=num_layers)

        self.indices_fc = nn.Sequential(nn.Linear(2 * hidden_dim * num_layers, 3),
                                        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        hidden = torch.cat(tuple([hidden[i] for i in range(2 * self.num_layers)]), dim=1)
        # hidden = hidden.squeeze(0)

        indices_output = self.indices_fc(hidden)

        return indices_output