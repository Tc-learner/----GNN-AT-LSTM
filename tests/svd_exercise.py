import numpy as np

# 定义矩阵A
A = np.array([[1, 5, 7],
              [2, -1, 3],
              [0, 1, -5]])

# 进行SVD分解
U, S, VT = np.linalg.svd(A)
print(S)
print(U)
# 计算B矩阵
B = np.dot(VT.T, U.T) / S[:, None]

print("B matrix:")
print(B)
