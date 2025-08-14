import torch

def my_softmax(u: torch.Tensor, dim: int = -1):
    u_max = torch.max(u, dim=dim, keepdim=True)
    exp_u = torch.exp(u - u_max)
    sum_exp_u = torch.sum(exp_u, dim=dim, keepdim=True)
    return exp_u / sum_exp_u

# expect q, k and v to be of shape [B, H, S, D]
def compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    qkt = torch.matmul(q, torch.transpose(k, -2, -1)) # resulting size is [B, H, S, S]
    qkt = qkt / (q.shape[-1] ** 0.5) # divide by sqrt(D)
    qkt = my_softmax(qkt) # size continues to be [B, H, S, S]
    # at this point, we have the attn_matrix in qkt
    return torch.matmul(qkt, v) # return a matrix of size [B, H, S, D]

qkv = torch.randn(3, batch_size, num_heads, seq_len, head_dim)
# we generated qkv in one shot. The size of the matrix will be [3, B, H, S, D]
q, k, v = qkv[:, ]
# at this point, we have three different matrices, each [B, H, S, D]

attn_matrix = compute_attention(q, k, v)

# TODO: do some tests to check if the attention matrix is correctly generated

print(attn_matrix.shape)
