import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtikelLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, num_layers):
        super(ArtikelLSTM, self).__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout, bidirectional=True,
                            num_layers=num_layers)

        self.indices_fc = nn.Sequential(nn.Linear(2 * hidden_dim * num_layers, 3),
                                        )

    def forward(self, x):
        embedded = F.one_hot(x, num_classes=self.vocab_size+1).float()
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        hidden = torch.cat(tuple([hidden[i] for i in range(2 * self.num_layers)]), dim=1)

        indices_output = self.indices_fc(hidden)

        return indices_output
