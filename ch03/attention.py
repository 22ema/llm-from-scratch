import torch.nn as nn
import torch

torch.manual_seed(123)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_query = nn.Parameter(torch.rand(d_in, d_out))
        self.w_key = nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.w_key
        queries = x @ self.w_query
        values = x @ self.w_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias)
        self.w_key = nn.Linear(d_in, d_out, bias)
        self.w_value = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, context_length, qkv_bias):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        attn_score = query @ key.transpose(1, 2)
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weight = torch.softmax(attn_score/key.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)
        context_vec = attn_weight @ value
        return context_vec


if __name__ == "__main__":
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)