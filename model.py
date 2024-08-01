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
            embed_dim: dimension of embeddings
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

# multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        """
        Args:
            embed_dim: dimension of embeddings
            num_heads: number of self attention heads
            dropout: dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # ensuring the embedding dimension is divisible by the number of heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        # dimension of each attention head's key, query, and value vectors
        self.single_head_dim
        return x.transpose(1, 2)= embed_dim // num_heads

        # Defining the weight matrices
        self.w_q = nn.Linear(embed_dim, embed_dim) # weight matrix for query
        self.w_k = nn.Linear(embed_dim, embed_dim) # weight matrix for key
        self.w_v = nn.Linear(embed_dim, embed_dim) # weight matrix for value
        self.w_o = nn.Linear(embed_dim, embed_dim) # weight matrix for output

    def split(self, x):
        """
        Args:
            x: vector of shape [batch_size, seq_len, embed_dim]
        Returns:
            x: vector of shape [batch_size, num_heads, seq_len, single_head_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.single_head_dim)
        # transpose to bring the head to the second dimension
        # final shape (batch_size, num_heads, seq_len, single_head_dim)
        return x.transpose(1, 2)

    def combine(self, x):
        """
        Args:
            x: vector of shape [batch_size, num_heads, seq_len, single_head_dim]
        Returns:
            x: vector of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, num_heads, seq_len, single_head_dim = x.size()
        embed_dim = num_heads * single_head_dim
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return x

    def scaled_dot_product_attention(self, q, k, v, attention_mask=None, key_padding_mask=None, dropout=None):
        """
        Args:
            q: vector of shape [batch_size, num_heads, query_sequence_length, single_head_dim]
            k: vector of shape [batch_size, num_heads, key_sequence_length, single_head_dim]
            v: vector of shape [batch_size, num_heads, key_sequence_length, single_head_dim]
            attention_mask: vector of shape [query_sequence_length, key_sequence_length]
            key_padding_mask : vector of shape [sequence_length, key_sequence_length]

        Returns:
            attention_values: vector of shape [batch_size, num_heads, seq_len, single_head_dim]
            attention_scores: vector of shape [batch_size, num_heads, seq_len, single_head_dim]
        """
        single_head_dim = k.size(-1)
        # tgt_len, src_len = q.size(-2), k.size(-2)

        # dot product Query with Key^T to compute similarity
        k_t = k.transpose(-2, -1) # transpose
        attention_scores = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product

        # apply attention masking (opt)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # apply key padding masking (opt)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # Broadcast over batch size, num heads
            attention_scores = attention_scores + key_padding_mask

        # softmax to make [0, 1] range
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # apply dropout to prevent overfitting (opt)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # multiply with Value
        attention_values = attention_scores @ v

        return attention_values, attention_scores

    def forward(self, q, k, v, attention_mask=None, key_padding_mask=None):
        """
        Args:
            q: vector of shape [batch_size, query_sequence_length, embed_dim]
            k: vector of shape [batch_size, key_sequence_length, embed_dim]
            v: vector of shape [batch_size, key_sequence_length, embed_dim]
            attention_mask: vector of shape [query_sequence_length, key_sequence_length]
            key_padding_mask : vector of shape [sequence_length, key_sequence_length]

        Returns:
            x: vector of shape [batch_size, seq_len, embed_dim]
        """
        # dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # split by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # scale dot product to compute similarity
        attention_values, attention_scores = self.scaled_dot_product_attention(q, k, v, attention_mask, key_padding_mask, self.dropout)
        self.attention_scores = attention_scores
        # combine and pass to linear layer
        x = self.combine(attention_values)
        x = self.w_o(x)

# feed forward network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff=2048, dropout=0.1):
        """
        Args:
            embed_dim: dimension of embeddings
            d_ff: dimension of feed forward
            dropout: dropout rate
        """
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
