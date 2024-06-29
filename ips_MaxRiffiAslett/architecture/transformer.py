# transformer.py
import math
import torch
from torch import nn
import torch.nn.functional as F



# b would define how many images are loaded and processed in each step.
# n_token would determine the initial number of patches extracted from each image.
# len_seq (M) would indicate the number of patches selected after iterative processing for the final aggregation and classification.
    
class ScaledDotProductAttention(nn.Module):
    ''' Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # Scale factor for the dot product
        self.dropout = nn.Dropout(attn_dropout)  # Dropout layer for attention weights

    def compute_attn(self, q, k):
        # Compute the scaled dot-product attention
        # When you call k.transpose(2, 3), it means that dimensions 2 and 3 of the tensor k are being swapped
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))  # Apply softmax and dropout
        return attn

    def forward(self, q, k, v):
        attn = self.compute_attn(q, k)  # Compute attention weights
        output = torch.matmul(attn, v)  # Apply attention weights to values
        return output

class MultiHeadCrossAttention(nn.Module):
    ''' Multi-head cross-attention module '''

    def __init__(self, n_token, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token = n_token  # M (sequence length)
        self.H = H  # Number of attention heads
        self.D_k = D_k  # Dimension of key vectors
        self.D_v = D_v  # Dimension of value vectors

        # Initialize learnable query parameter with shape (1, M, D)
        self.q = nn.Parameter(torch.empty((1, n_token, D)))
        q_init_val = math.sqrt(1 / D_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)

        # Define linear layers for projecting input to queries, keys, and values
        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc = nn.Linear(H * D_v, D, bias=False)  # Final linear layer to combine heads

        self.attention = ScaledDotProductAttention(
            temperature=D_k ** 0.5,  # Scale factor for attention
            attn_dropout=attn_dropout  # Dropout for attention weights
        )

        self.dropout = nn.Dropout(dropout)  # Dropout for final output
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)  # Layer normalization
        self.attn_maps = None  # To store attention maps for diversity loss

    def get_attn(self, x):
        # Compute attention maps (without applying them to values)
        D_k, H, n_token = self.D_k, self.H, self.n_token
        B, len_seq = x.shape[:2] # B represents the batch size

        # Project inputs to queries and keys
        # The .view() function takes one or more integers as arguments, which represent the desired shape of the output tensor.
        q = self.q_w(self.q).view(1, n_token, H, D_k)  # (1, n_token, H * D_k) -> (1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)  # (B, len_seq, H * D_k) -> (B, len_seq, H, D_k)

        # Transpose to shape (B, H, len_seq, D_k) for attention calculation
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (1, H, n_token, D_k) and (B, H, len_seq, D_k)
        attn = self.attention.compute_attn(q, k)  # Compute attention weights

        return attn


# b would define how many images are loaded and processed in each step.
# n_token would determine the initial number of patches extracted from each image.
# len_seq would indicate the number of patches selected after iterative processing for the final aggregation and classification.
    
    def forward(self, x):
        D_k, D_v, H, n_token = self.D_k, self.D_v, self.H, self.n_token
        B, len_seq = x.shape[:2]

        # Project and reshape queries, keys, and values
        q = self.q_w(self.q).view(1, n_token, H, D_k)  # (1, n_token, H * D_k) -> (1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)  # (B, M, H * D_k) -> (B, M, H, D_k)
        v = self.v_w(x).view(B, len_seq, H, D_v)  # (B, M, H * D_v) -> (B, M, H, D_v)

        # Transpose for attention dot product: (B, H, len_seq, D_k or D_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, M, D_k), (B, H, M, D_k), (B, H, M, D_v)
        attn = self.attention(q, k, v)  # Apply scaled dot-product attention
        x = torch.matmul(attn, v)  # Apply attention weights to values -> (B, H, M, D_v)

        # Store attention maps for diversity loss
        self.attn_maps = attn

        # Transpose again: (B, n_token, H, D_v), then concatenate heads: (B, n_token, H * D_v)
        x = x.transpose(1, 2).contiguous().view(B, n_token, -1)  # (B, n_token, H * D_v)
        x = self.dropout(self.fc(x))  # Apply final linear layer and dropout -> (B, n_token, D)
        x += self.q  # Add residual connection -> (B, n_token, D)
        x = self.layer_norm(x)  # Apply layer normalization -> (B, n_token, D)

        return x

    def compute_diversity_loss(self):
        # Compute the diversity loss based on the stored attention maps
        if self.attn_maps is None:
            raise ValueError("Attention maps not computed. Ensure forward pass is run before computing diversity loss.")

        M, N = self.attn_maps.shape[1], self.attn_maps.shape[2]  # M is the number of heads, N is the sequence length
        diversity_loss = 0
        for i in range(M):
            for j in range(i + 1, M):
                # Compute pairwise cosine similarity and sum it up
                diversity_loss += F.cosine_similarity(self.attn_maps[:, i, :], self.attn_maps[:, j, :], dim=-1).mean()

        # Normalize by the number of pairs
        diversity_loss = (2 / (M * (M - 1))) * diversity_loss
        return diversity_loss

class MLP(nn.Module):
    ''' MLP consisting of two feed-forward layers '''

    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        
        x += residual
        x = self.layer_norm(x)

        return x

class Transformer(nn.Module):
    """ Cross-attention based transformer module """

    def __init__(self, n_token, H, D, D_k, D_v, D_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.crs_attn = MultiHeadCrossAttention(n_token, H, D, D_k, D_v, attn_dropout=attn_dropout, dropout=dropout)
        self.mlp = MLP(D, D_inner, dropout=dropout)
    
    def get_scores(self, x):

        attn = self.crs_attn.get_attn(x)
        # Average scores over heads and tasks
        # Average over tasks is only required for multi-task learning (mnist).
        return attn.mean(dim=1).transpose(1, 2).mean(-1)

    def forward(self, x):

        return self.mlp(self.crs_attn(x))
