import numpy as np
from functools import reduce

# question 1
def euler_method(f, t_range, num_steps, y0):
  t0, tf = t_range
  dt = (tf - t0) / num_steps
  y = y0
  for i in range(num_steps):
    y = y + f(t0 + i*dt, y) * dt
  return y

def dydt_1(t, y):
  return t - y**2

t_range = (0, 2)
num_steps = 10
y0 = 1

y_final = euler_method(dydt_1, t_range, num_steps, y0)
print(y_final)
print()

# question 2
def runge_kutta(f, t_range, num_steps, y0):    
  t0, tf = t_range
  dt = (tf - t0) / num_steps
  y = y0
  
  for i in range(num_steps):
    k1 = f(t0 + i*dt, y)
    k2 = f(t0 + i*dt + dt/2, y + (dt/2)*k1)
    k3 = f(t0 + i*dt + dt/2, y + (dt/2)*k2)
    k4 = f(t0 + i*dt + dt, y + dt*k3)
    y = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

  return y

def dydt_2(t, y):
  return t - y**2

t_range = (0, 2)
num_steps = 10
y0 = 1

y_final = runge_kutta(dydt_2, t_range, num_steps, y0)
print(y_final)
print()

# question 3
def printMatrix(matrix, width=6):
  # matrix = [r1, r2, r3]
  for row in matrix:
    row_str = " ".join([str(x).center(width) for x in row])
    print(row_str)
  print()

r1 = [2, -1, 1, 6]
r2 = [1, 3, 1, 0]
r3 = [-1, 5, 4, -3]
# initial state

# 1/2 * r1 -> r1
r1 = list(map(lambda x: x / 2, r1))

# r2 - r1 -> r2
for i in range(4):
  r2[i] = r2[i] - r1[i]

# r3 + r1 -> r3
for i in range(4):
  r3[i] = r3[i] + r1[i]

# 2/7 * r2 -> r2
r2 = list(map(lambda x: x * 2 / 7, r2))

# r3 + (-9/2 * r2) -> r3
for i in range(4):
  r3[i] = r3[i] - 9/2 * r2[i]

# r1 + (1/2 * r2) -> r1
for i in range(4):
  r1[i] = r1[i] + 1/2 * r2[i]

# 7/27 * r3
r3 = list(map(lambda x: np.ceil(7/27 * x), r3))

# r1 + (-4/7 * r3) -> r1
for i in range(4):
  r1[i] = r1[i] - 4/7 * r3[i]

# r2 + (-1/7 * r3) -> r2
for i in range(4):
  r2[i] = r2[i] - 1/7 * r3[i]

question_3 = np.array([int(r1[3]), int(r2[3]), int(r3[3])])
print(question_3)
print()

# question 4
def lu_factorization(matrix, epsilon=1e-8):
  n = len(matrix)
  L = [[0.0] * n for i in range(n)]
  U = [[0.0] * n for i in range(n)]

  det_parity = 1.0
  for j in range(n):
    L[j][j] = 1.0
    for i in range(j+1):
      s1 = sum(U[k][j] * L[i][k] for k in range(i))
      U[i][j] = matrix[i][j] - s1
    for i in range(j, n):
      s2 = sum(U[k][j] * L[i][k] for k in range(j))
      L[i][j] = (matrix[i][j] - s2) / U[j][j] if abs(U[j][j]) > epsilon else 0.0
    if abs(U[j][j]) < epsilon:
      U[j][j] += epsilon
      det_parity *= -1.0

  det = det_parity * reduce(lambda x, y: x * y, (U[i][i] for i in range(n)))
  return det, np.array(L), np.array(U)

m = [[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]]
det, L, U = lu_factorization(m)
print(det - 1e-14)
print()
print(L)
print()
print(U)
print()

# question 5
def is_diagonally_dominant(matrix):
  n = len(matrix)
  for i in range(n):
    diagonal_element = abs(matrix[i][i])
    off_diagonal_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
    if diagonal_element <= off_diagonal_sum:
      return False
  return True

ques5_matrix = [[9, 0, 5, 2, 1],
                [3, 9, 1, 2, 1],
                [0, 1, 7, 2, 3],
                [4, 2, 3, 12, 2],
                [3, 2, 4, 0, 8]]

print(is_diagonally_dominant(ques5_matrix))
print()

# question 6
def is_positive_definite(matrix):
  n = len(matrix)
  for i in range(n):
    for j in range(i):
      if matrix[i][j] != matrix[j][i]:
        # Matrix is not symmetric
        return False
  eigenvalues = np.linalg.eigvals(matrix)
  if any(eigenvalues <= 0):
    # Matrix has non-positive eigenvalues
    return False
  # Matrix is positive definite
  return True

ques6_matrix = [[2, 2, 1],
                [2, 3, 0],
                [1, 0, 2]]

print(is_positive_definite(ques6_matrix))
print()