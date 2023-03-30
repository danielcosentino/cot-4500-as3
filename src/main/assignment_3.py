import numpy as np

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
def printMatrix(r1, r2, r3, width=6):
  matrix = [r1, r2, r3]
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