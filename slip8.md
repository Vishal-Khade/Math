
# Slip No. 8

## Q.1. Attempt any two of the following. [10]

### 1. Use Python code to find \( a + c \), \( ab \), \( cd \), \( \frac{a}{b} \), and \( a(b + c) \), where \( a = 5 \), \( b = 7 \), \( c = 9 \), \( d = 11 \).

```python
a = 5
b = 7
c = 9
d = 11

# Calculating the required expressions
sum_ac = a + c
product_ab = a * b
product_cd = c * d
division_ab = a / b
expression = a * (b + c)

print("a + c =", sum_ac)
print("ab =", product_ab)
print("cd =", product_cd)
print("a / b =", division_ab)
print("a(b + c) =", expression)
```

### 2. The following two statements using the ‘+’ string operation in Python.

```python
# a. Concatenating strings
string1 = "India Won"
string2 = " World Cup"
combined1 = string1 + string2
print("Combined String 1:", combined1)

# b. Concatenating strings
string1 = "God"
string2 = " is Great"
combined2 = string1 + string2
print("Combined String 2:", combined2)
```

### 3. Write Python code to find the area and circumference of a circle with radius 14.

```python
import math

radius = 14

# Area of the circle
area = math.pi * radius**2
# Circumference of the circle
circumference = 2 * math.pi * radius

print("Area of the circle:", area)
print("Circumference of the circle:", circumference)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, logically verify associativity of matrices with respect to matrix addition (use proper matrices).

```python
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])

# Calculate A + (B + C) and (A + B) + C
lhs = A + (B + C)
rhs = (A + B) + C

print("A + (B + C):\n", lhs)
print("(A + B) + C:\n", rhs)
print("Associativity holds:", np.array_equal(lhs, rhs))
```

### 2. Write Python code to generate 10 terms of the Fibonacci Sequence using a loop.

```python
# Fibonacci Sequence
n_terms = 10
fib_sequence = [0, 1]

for i in range(2, n_terms):
    next_term = fib_sequence[-1] + fib_sequence[-2]
    fib_sequence.append(next_term)

print("Fibonacci Sequence (first 10 terms):", fib_sequence)
```

### 3. Using Python code, find the determinant and inverse of the matrix if it exists.
\[ A =
\begin{bmatrix}
4 & 2 & 2 \\
2 & 4 & 2 \\
2 & 2 & 4
\end{bmatrix}
\]

```python
A = np.array([[4, 2, 2],
              [2, 4, 2],
              [2, 2, 4]])

# Calculate determinant
determinant = np.linalg.det(A)

# Calculate inverse if determinant is not zero
if determinant != 0:
    inverse = np.linalg.inv(A)
else:
    inverse = None

print("Determinant of A:", determinant)
print("Inverse of A:\n", inverse)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral \( \int_0^1 (1 + x^2) \, dx \) using Simpson’s \( \frac{1}{3} \) rule (n=6).

```python
def simpson_13(a, b, n, f):
    h = (b - a) / n
    I = f(a) + f(b)
    
    for i in range(1, n):
        k = a + i * h
        if i % 2 == 0:
            I += 2 * f(k)
        else:
            I += 4 * f(k)
    
    I *= h / 3
    return I

# Function to integrate
def integrand(x):
    return 1 + x**2

a = 0
b = 1
n = 6
integral_value = simpson_13(a, b, n, integrand)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to evaluate the fourth order forward difference of the given data.

```python
import numpy as np

# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([41, 62, 65, 50, 17])

# Function to calculate forward differences
def forward_difference(y):
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table

# Calculate forward differences
diff_table = forward_difference(y)
print("Forward Difference Table:\n", diff_table)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of \( x^3 - 2x - 5 = 0 \) in \([2,3]\) using the Regula-falsi method.

```python
def regula_falsi(a, b, f, n):
    for i in range(n):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        print(f"Iteration {i+1}, c = {c}, f(c) = {f(c)}")
    return c

def f(x):
    return x**3 - 2*x - 5

a = 2
b = 3
n = 10
root = regula_falsi(a, b, f, n)
print("Approximate real root:", root)
```

### 2. Write a Python program to estimate the value of the integral \( \int_2^4 (2x^2 - 4x + 1) \, dx \) using the Trapezoidal rule (n=5).

```python
def trapezoidal(a, b, n, f):
    h = (b - a) / n
    I = (f(a) + f(b)) / 2
    
    for i in range(1, n):
        I += f(a + i * h)
    
    I *= h
    return I

def integrand(x):
    return 2 * x**2 - 4 * x + 1

a = 2
b = 4
n = 5
integral_value = trapezoidal(a, b, n, integrand)

print("Estimated value of the integral:", integral_value)
```

