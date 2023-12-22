import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):

    def __init__(self, num_feats, conv_kernel_size, embedding_size, num_layers,
        dropout=0.0):
     
        super(ConvLSTM, self).__init__()

        self.num_feats = num_feats
        self.conv_kernel_size = conv_kernel_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv1 = nn.Conv1d(
            in_channels=num_feats,
            out_channels=embedding_size,
            kernel_size=conv_kernel_size,
        )

        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 1)
       
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(self.embedding_size)

        self.o_proj = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        y = torch.permute(x, [0, 2, 1]) 
        y = F.pad(y, (self.conv_kernel_size,0), 'replicate') 
        y = self.conv1(y)
        y = self.relu1(y)
        y = self.pool(y)

        y = torch.permute(y, [0, 2, 1])
        y, (hn, cn) = self.lstm(y)
        y = self.o_proj(y)

        return self.sigmoid(y).squeeze(2)

if __name__ == '__main__':

    bs, seq, feats = 128, 420, 8
    embed = 69

    x = torch.randn(bs, seq, feats)
    model = ConvLSTM(feats, 5, embed, 1)
    y = model(x)

    print(f'shape should be [{bs}, {seq}, 1]')
    print(f'shape is {y.shape}')
