import torch
# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
ans = torch.matmul(tensor1, tensor2)
ans_size = ans.size()
# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
ans = torch.matmul(tensor1, tensor2)
ans_size = ans.size()
# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
ans = torch.matmul(tensor1, tensor2)
ans_size = ans.size()
# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
ans = torch.matmul(tensor1, tensor2)
ans_size = ans.size()
# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
ans = torch.matmul(tensor1, tensor2)
ans_size = ans.size()
print(ans_size)
