import torch
import torch.nn as nn

input = torch.tensor([1.0, 2.0])

tensor = torch.rand(2, 2)

query_linear = nn.Linear(2, 2, bias=False)
query_param = nn.Parameter(query_linear.weight.T)

print("LINEAR: ", query_linear.weight)
print("Param: ", query_param)

print("param result: ", input @ query_param)
print("linear result: ", query_linear(input))



