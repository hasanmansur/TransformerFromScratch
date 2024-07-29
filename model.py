# importing required libraries
import torch.nn as nn

# input embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # embedding layer to convert indices to dense embeddings of shape [vocab_size, embed_dim]
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector of shape [batch_size, seq_len, embed_dim]
        """
        out = self.embed(x) * math.sqrt(self.embed_dim)
        return out

# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embed_dim, dropout=0.1):
        """
        Args:
            max_seq_len: max length of input sequence
            embed_dim: demension of embedding
            dropout: dropout rate
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # creating a positional encoding matrix of shape [seq_len, embed_dim] filled with zeros
        pe = torch.zeros(max_seq_len, self.embed_dim)
        # applying encoding rules on the positional encoding matrix 'pe'
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        # adding an extra dimension at the beginning of pe matrix for batch handling
        pe = pe.unsqueeze(0)
        # registering 'pe' as buffer so that it is not considered as a model parameter
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector of shape [batch_size, seq_len, embed_dim]
        Returns:
            x: output vector of shape [batch_size, seq_len, embed_dim]
        """

        # making embeddings relatively larger so that the positional encoding
        # becomes relatively smaller. This means the original meaning in the embedding
        # vector won’t be lost when we add them together.
        x = x * math.sqrt(self.embed_dim)
        # adding positional encoding to the input vector x
        seq_len = x.size(1)
        x = x + (self.pe[:, :seq_len, :]).requires_grad_(False)
        # dropout for regularization
        return self.dropout(x)

# ------ transformer encoder block --------------

# multihead attention

# feed forward layer
class FeedForward(nn.Module):
    """
    Args:
        embed_dim: demension of embedding
        d_ff: dimension of feed forward
        dropout: dropout rate
    """

    def __init__(self, embed_dim, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff) # first linear transformation
        self.fc2 = nn.Linear(d_ff, embed_dim) # second linear transformation
        self.relu = nn.ReLU() # ReLU activation function
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

    def forward(self, x):
        """
        Args:
            x: input vector of shape [batch_size, seq_len, embed_dim]
        Returns:
            x: output vector of shape [batch_size, seq_len, embed_dim]
        """
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# transformer encoder

# transformer decoder block

# transformer decoder

# transformer
