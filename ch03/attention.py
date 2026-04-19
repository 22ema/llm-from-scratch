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

if __name__ == "__main__":
    # x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2
    #
    # w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    # w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    # w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    #
    # queries = inputs @ w_query
    # keys = inputs @ w_key
    # values = inputs @ w_value
    #
    # attn_score = queries @ keys.T
    # print(attn_score)
    #
    # d_k = keys.shape[-1]
    # attn_weight = torch.softmax(attn_score / d_k**0.5, dim=-1)
    # print(attn_weight)
    #
    # context_vec = attn_weight @ values
    # print(context_vec)

    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
