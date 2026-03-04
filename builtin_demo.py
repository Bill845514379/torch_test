import torch

# 1. 张量创建
def test_1():
    # 创建一个全零张量
    x = torch.zeros(3, 4)
    print("zeros:\n", x)

    # 创建一个随机张量（标准正态分布）
    y = torch.randn(2, 3)
    print("randn:\n", y)

    # 从 Python 列表创建
    z = torch.tensor([[1, 2], [3, 4]])
    print("from list:\n", z)

# 基本数学运算（element-wise）
def test_2():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("a + b =", a + b)          # 加法
    print("a * b =", a * b)          # 逐元素乘法
    print("a ** 2 =", a ** 2)        # 幂运算
    print("torch.sin(a) =", torch.sin(a))  # 三角函数

# 矩阵运算
def test_3():
    A = torch.tensor([[1., 2.], [3., 4.]])
    B = torch.tensor([[5., 6.], [7., 8.]])

    # 矩阵乘法
    C = torch.mm(A, B)   # 或 A @ B
    print("Matrix multiplication:\n", C)

    # 转置
    print("A transpose:\n", A.t())

# 形状操作
def test_4():
    x = torch.arange(12).reshape(3, 4)
    print("Original:\n", x)

    # reshape / view
    y = x.view(2, 6)
    print("Reshaped:\n", y)

    # 转置
    z = x.transpose(0, 1)
    print("Transposed:\n", z)

    # 展平
    flat = x.flatten()
    print("Flattened:", flat)

#  索引与切片
def test_5():
    x = torch.randn(4, 5)
    print("x:\n", x)

    # 获取第一行
    print("First row:", x[0])

    # 获取前两列
    print("First two columns:\n", x[:, :2])

    # 条件索引（masking）
    mask = x > 0
    print("Positive elements:", x[mask])

# 聚合操作
def test_6():
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    print("Sum all:", x.sum())
    print("Sum along dim=0:", x.sum(dim=0))  # 列方向求和
    print("Mean along dim=1:", x.mean(dim=1))  # 行方向平均
    print("Max:", x.max())
    print("Argmax (global):", x.argmax())  # 返回 flatten 后的索引


# 自动求导（Autograd）——PyTorch 的核心特性
def test_7():
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print("dy/dx at x=2:", x.grad)  # 应该输出 2*2 + 3 = 7


if __name__ == '__main__':
    test_7()
