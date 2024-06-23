import torch
import math
# 查询张量 (batch_size, query_length, query_dim)
query = torch.randn(2, 4, 64)

# 键张量 (batch_size, key_length, key_dim)
key = torch.randn(2, 6, 64)

# 值张量 (batch_size, key_length, value_dim)
value = torch.randn(2, 6, 128)

# 掩码张量 (batch_size, 1, key_length)
mask = torch.randint(0, 2, (2, 1, 6), dtype=torch.bool)

def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=3):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print(scores.shape)
    #if mask is not None:
        #scores = scores.masked_fill(mask == 0, float('-inf'))
    print(scores)
    selected_scores, idx = scores.topk(topk)
    print(selected_scores.shape,idx.shape)
    print(selected_scores,idx)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn

print(value.unsqueeze(2).shape)
print(memory_querying_responding(query, key, value, mask=mask))
