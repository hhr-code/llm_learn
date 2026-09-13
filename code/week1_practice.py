"""第 1 周练习：张量操作、自动微分、MPS 加速。
运行方式: conda activate pytorch_env && python code/week1_practice.py
"""
import time

import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"MPS 可用: {torch.backends.mps.is_available()}")

# ---------- 练习 1：张量操作热身 ----------
x = torch.randn(3, 4)
print("shape:", x.shape)
print("reshape:", x.reshape(4, 3).shape)
print("矩阵乘法:", (x @ x.T).shape)

# 进阶（周一周二自己加）：三维张量、squeeze/unsqueeze、广播
y = torch.randn(2, 3, 4)
print("三维张量 sum(dim=1) 形状:", y.sum(dim=1).shape)

# ---------- 练习 2：自动微分验证 ----------
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2 * x + 1
y.backward()
print(f"练习2 x.grad = {x.grad.item()} (期望 8.0)")
assert abs(x.grad.item() - 8.0) < 1e-6

# ---------- 练习 3：MPS vs CPU 性能对比 ----------
def bench(device: str, n: int = 1000, iters: int = 10):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    # 先跑一次预热，避免首次数值影响计时
    z = a @ b
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        z = a @ b
    if device == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - start) / iters, z

cpu_time, _ = bench("cpu")
if torch.backends.mps.is_available():
    mps_time, z = bench("mps")
    print(f"CPU: {cpu_time*1000:.1f} ms/次, MPS: {mps_time*1000:.1f} ms/次, 加速 {cpu_time/mps_time:.1f}x")
    print(f"练习3 结果 device = {z.device}")
    print("注：矩阵太小（1.8ms vs 1.4ms）时 GPU 加速不明显——数据搬运开销占比大，这本身就是个值得记住的现象")
else:
    print(f"CPU: {cpu_time*1000:.1f} ms/次")
