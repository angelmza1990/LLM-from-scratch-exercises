import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)”
)

print("inputs:\n", inputs)
x = torch.tensor([1,2,3])
w = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = x @ w.T
print('w', w)
print('w.t', w.T)
print("result of x @ w.T:\n", result)
x_2 = inputs[1]
print(w)
d_in = inputs.shape[1] # 3
d_out = 2
print('shapes',inputs.shape)
print("x_2:\n", x_2)

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print("W_query:\n", W_query)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("query_2: \n", query_2)

keys = inputs @ W_key
values = inputs @ W_value

print("keys:\n", keys)
print("values:\n", values)


keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("attn_score_22:\n", attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("attn_scores_2:\n", attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn_weights_2:\n", attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print("context_vec_2:\n", context_vec_2)
