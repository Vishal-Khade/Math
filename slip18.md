
# Slip No. 18

## Q.1. Attempt any two of the following. [10]

### 1. Use Python code to find the minimum value from the given numbers: 16, 3, 5, 48, 2, 4, 5, 6, 78, 12, 5, 6, 24.

```python
# List of numbers
numbers = [16, 3, 5, 48, 2, 4, 5, 6, 78, 12, 5, 6, 24]

# Finding the minimum value
min_value = min(numbers)
print("Minimum value:", min_value)
```

### 2. Use Python code to find the hypotenuse of a triangle whose sides are 12 and 5.

```python
import math

# Sides of the triangle
a = 12
b = 5

# Calculating the hypotenuse
hypotenuse = math.sqrt(a**2 + b**2)
print("Hypotenuse:", hypotenuse)
```

### 3. Use Python code to remove all digits after the decimal of the given number 125312.3142.

```python
# Given number
number = 125312.3142

# Removing digits after the decimal
integer_part = int(number)
print("Integer part:", integer_part)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, find the below matrices, where 

\[
A = \begin{pmatrix} 2 & 4 \\ 4 & 3 \end{pmatrix}
\]
and 
\[
B = \begin{pmatrix} 4 & 3 \\ 5 & 4 \end{pmatrix}
\]

(a) \( A + B \)

(b) \( A^T \)

(c) \( A^{-1} \)

```python
import numpy as np

# Define matrices A and B
A = np.array([[2, 4],
              [4, 3]])

B = np.array([[4, 3],
              [5, 4]])

# (a) A + B
sum_matrix = A + B
print("A + B:\n", sum_matrix)

# (b) A^T (Transpose of A)
transpose_A = A.T
print("Transpose of A:\n", transpose_A)

# (c) A^{-1} (Inverse of A)
inverse_A = np.linalg.inv(A)
print("Inverse of A:\n", inverse_A)
```

### 2. Use a while loop in Python to find the sum of the first twenty natural numbers.

```python
# Initialize variables
sum_natural_numbers = 0
n = 1  # Starting from 1

# Using while loop to find the sum of the first 20 natural numbers
while n <= 20:
    sum_natural_numbers += n
    n += 1

print("Sum of the first twenty natural numbers:", sum_natural_numbers)
```

### 3. Write Python program to diagonalize the matrix:

\[
\begin{pmatrix} 3 & -2 \\ 6 & -4 \end{pmatrix}
\]

and find matrices \( P \) and \( D \).

```python
# Define the matrix
matrix = np.array([[3, -2],
                   [6, -4]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Diagonal matrix D
D = np.diag(eigenvalues)

# Matrix P of eigenvectors
P = eigenvectors

print("Diagonal Matrix D:\n", D)
print("Matrix P (Eigenvectors):\n", P)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral \( R_{1}^{3} \frac{1}{x} \, dx \) using Simpson’s \( \left(\frac{1}{3}\right) \) rule (n=8).

```python
def f(x):
    return 1 / x

def simpson_rule(a, b, n):
    if n % 2 == 1:  # n must be even
        raise ValueError("n must be an even number.")
        
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

# Parameters
a = 1
b = 3
n = 8
integral_value = simpson_rule(a, b, n)

print("Estimated value of the integral R_1^3 (1/x) dx:", integral_value)
```

### 2. Write Python program to evaluate the interpolate value \( f(2.9) \) of the given data.

\[
\begin{align*}
x: & \quad 1 \quad 2 \quad 3 \quad 4 \\
Y = f(x): & \quad 11 \quad 9 \quad 27 \quad 64
\end{align*}
\]

```python
# Lagrange interpolation function
def lagrange_interpolation(x, y, x_target):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_target - x[j]) / (x[i] - x[j])
        result += term
    return result

# Known points
x_values = [1, 2, 3, 4]
y_values = [11, 9, 27, 64]

# Target value
x_target = 2.9
f_2_9 = lagrange_interpolation(x_values, y_values, x_target)

print(f"Interpolated value f(2.9) = {f_2_9}")
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of \( x^3 - 5x - 9 = 0 \) in [2, 3] using the Regula-Falsi method.

```python
def f(x):
    return x**3 - 5*x - 9

root_regula_falsi = regula_falsi(f, 2, 3)
print("Approximate real root using Regula-Falsi method:", root_regula_falsi)
```

### 2. Write Python program to estimate the value of the integral \( R_{0}^{1} \cos(x) \, dx \) using the Trapezoidal rule (n=5).

```python
import numpy as np

def g(x):
    return np.cos(x)

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    I = (g(a) + g(b)) / 2

    for i in range(1, n):
        k = a + i * h
        I += g(k)

    I *= h
    return I

# Parameters
a = 0
b = 1
n = 5
integral_value_cos = trapezoidal_rule(a, b, n)

print("Estimated value of the integral R_0^1 cos(x) dx:", integral_value_cos)
```
```

