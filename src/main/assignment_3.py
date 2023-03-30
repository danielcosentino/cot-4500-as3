import numpy as np
from decimal import Decimal
                 #

val = "010000000111111010111001"

# val = "010000000011101110010001"

double_bias = 1023


q1_sign = int(val[0:1], 2)

q1_exponent = int(val[1:12], 2)

q1_mantissa = val[12:]


q1_fraction = 0
i = 0

for digit in q1_mantissa:

  i -= 1

  if digit == '1':

    q1_fraction += pow(2, i)

q1_power = q1_exponent - double_bias

q1_result = pow(-1, q1_sign) * pow(2, q1_power) * (1 + q1_fraction)
# question 1

print("{0:.4f}".format(q1_result))
print()


# question 2

q2_result = float(int(q1_result))

print("{0:.1f}".format(q2_result))
print()


# question 3

q3_result = float(int(q1_result + 0.5))

print("{0:.1f}".format(q3_result))
print()


# question 4

abs_error = Decimal(abs((q1_result) - (q3_result)))


print(abs_error)

rel_error = (abs_error / abs(Decimal(q1_result)))
print(rel_error)
print()


# question 5

# 1/(n+1) ^ 3 < 10^-4

# (n+1) ^ 3 > 10^4

# n + 1 > 10^(4/3)

# n > 10^(4/3) - 1

print(int(pow(10, 4/3)))
print()

# question 6

# Determine the number of iterations necessary to solve f(x) = x^3 + 4x^2 – 10 = 0 with

# accuracy 10^-4 using a = 4 and b = 7

# a: Bisection


def f(x):
  return pow(x,3) + 4 * pow(x, 2) - 10

tol = pow(10, -4)
left = -4
right = 7

max = 1000
i = 0

while (abs(right - left) > tol and i < max):
  i += 1
  p = (left + right) / 2
  # print("at step " + str(i) + ", f(" + str(p) + ") = " + str(f(p)))
  if (f(left) < 0 and f(p) > 0) or (f(left) > 0 and f(p) < 0):
    right = p
  else:
    left = p
print(i)

print()

# Determine the number of iterations necessary to solve f(x) = x^3 + 4x^2 – 10 = 0
def fp(x):
  return 3 * pow(x, 2) + 8 * x

# b: Raphson

# p_prev = (7+(-4))/2

p_prev = -4
p_next = 0

i = 1
while (i <= max):
  if (fp(p_prev) != 0):
    p_next = p_prev - f(p_prev) / fp(p_prev)
    if (abs(p_next - p_prev) < tol):
      print(i)
      break
    i += 1
    p_prev = p_next
  else:
    break
print()




        