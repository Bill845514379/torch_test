import torch

def scan(fn, init, xs):
    """
    模拟 scan 函数。
    :param fn: (state, x) -> new_state
    :param init: 初始状态
    : :param xs: 输入序列，shape (T, ...)
    :return: states: 所有中间状态，shape (T, ...)
    """
    states = []
    state = init
    for x in xs:
        state = fn(state, x)
        states.append(state)
    return torch.stack(states)

# 示例：累积加法
xs = torch.tensor([1.0, 2.0, 3.0, 4.0])  # shape: (4,)
init = torch.tensor(0.0)

def add(state, x):
    return state + x

result = scan(add, init, xs)
print(result)  # 输出: tensor([1., 3., 6., 10.])
