import torch
import torch.nn as nn

class GruModelGeneral(nn.Module):
    """ 
    Defines the GRU model for classification.
    """
    def __init__(self, in_features, num_classes, num_layers=1, hidden_size=64,
                 embed_size=64, dropout=0.8):
        super(GruModelGeneral, self).__init__()
        
        self.in_features = in_features
        if self.in_features == 2:
            self.embed = torch.nn.Linear(in_features, embed_size)
        else:
            self.embed = torch.nn.Embedding(num_embeddings=in_features,
                                        embedding_dim=embed_size)     
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size=embed_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.name = 'GruModelGeneral'
        self.dropout1 = nn.Dropout(0.5)

    def initHidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, h):
        y = self.embed(x)
        y = self.dropout1(y)
        y, h = self.gru(y, h)
        y = self.fc(y)
        return y, h

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        # Input dim
        self.input_dim = input_dim
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Output size
        self.output_dim = output_dim

        # batch_first=True puts batch dimension in the beginning
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out
