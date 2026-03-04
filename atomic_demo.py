import torch

# CPU 示例
values = torch.tensor([1.0, 2.0, 3.0, 4.0])
indices = torch.tensor([0, 1, 0, 1])  # 两个值要加到位置 0 和 1
output = torch.zeros(2)

output.scatter_add_(0, indices, values)
print(output)  # tensor([4., 6.])  => 1+3=4, 2+4=6
