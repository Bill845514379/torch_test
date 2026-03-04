import torch

x = torch.tensor([[1, 2],
                  [3, 4]])  # shape: (2, 2)

# 沿 dim0 重复 2 次，dim1 重复 3 次
y = torch.tile(x, (2, 3))
print(y)
# 输出:
# tensor([[1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4],
#         [1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4]])
# shape: (4, 6)
