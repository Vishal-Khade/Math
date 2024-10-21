
# Slip No. 23

## Q.1. Attempt any two of the following. [10]

### 1. Write Python program to find the sum of first n natural numbers.

```python
# Find the sum of first n natural numbers
n = int(input("Enter a natural number n: "))
sum_n = n * (n + 1) // 2
print("Sum of first", n, "natural numbers is:", sum_n)
```

### 2. Write Python code to print all integers between 1 to 100 that are divisible by 3 and 7.

```python
# Print all integers between 1 to 100 that are divisible by 3 and 7
divisible_by_3_and_7 = [i for i in range(1, 101) if i % 3 == 0 and i % 7 == 0]
print("Integers between 1 to 100 that are divisible by 3 and 7:", divisible_by_3_and_7)
```

### 3. Write Python code to print all integers between 1 to n, which are relatively prime to n.

```python
from math import gcd

# Find all integers between 1 to n that are relatively prime to n
n = int(input("Enter a natural number n: "))
relatively_prime = [i for i in range(1, n + 1) if gcd(i, n) == 1]
print("Integers between 1 to", n, "that are relatively prime to", n, ":", relatively_prime)
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python code to find determinant, transpose and inverse of matrix 

\[
A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 5 & 7 \\ 4 & 9 & 11 \end{pmatrix}
\]

```python
from sympy import Matrix

# Define the matrix A
A = Matrix([[1, 2, 3],
             [2, 5, 7],
             [4, 9, 11]])

# Calculate the determinant
determinant_A = A.det()
print("Determinant of matrix A:", determinant_A)

# Calculate the transpose
transpose_A = A.transpose()
print("Transpose of matrix A:\n", transpose_A)

# Calculate the inverse
inverse_A = A.inv()
print("Inverse of matrix A:\n", inverse_A)
```

### 2. Write Python program to find the roots of the quadratic equation \( ax^2 + bx + c = 0 \).

```python
import cmath

# Function to find roots of the quadratic equation
def find_roots(a, b, c):
    discriminant = b**2 - 4*a*c
    root1 = (-b + cmath.sqrt(discriminant)) / (2*a)
    root2 = (-b - cmath.sqrt(discriminant)) / (2*a)
    return root1, root2

# Input coefficients
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))

roots = find_roots(a, b, c)
print("Roots of the quadratic equation are:", roots)
```

### 3. Using Python solve the following system of equations using LU – Factorization method:

\[
\begin{align*}
3x - 7y - 2z & = -7 \\
-3x + 5y + z & = 5 \\
6x - 4y & = 2
\end{align*}
\]

```python
import numpy as np
from scipy.linalg import lu

# Define the coefficient matrix and the right-hand side vector
A = np.array([[3, -7, -2],
              [-3, 5, 1],
              [6, -4, 0]])

b = np.array([-7, 5, 2])

# LU Factorization
P, L, U = lu(A)

# Solve the equation Ax = b
y = np.linalg.solve(L, b)  # Solve Ly = b
x = np.linalg.solve(U, y)  # Solve Ux = y

print("Solution of the system of equations (x, y, z):", x)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral 

\[
\int_1^3 \frac{1}{x} \, dx
\]

by using Simpson’s (1/3) rule (n=8).

```python
import numpy as np

# Simpson's 1/3 Rule Implementation
def simpson_13_rule(f, a, b, n):
    if n % 2 == 1:
        raise ValueError("n must be an even integer.")
    
    h = (b - a) / n
    integral = f(a) + f(b)
    
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        integral += 2 * f(a + i * h)
        
    return integral * h / 3

# Define the function
def f(x):
    return 1/x

# Estimate the integral from 1 to 3 with n = 8
a, b, n = 1, 3, 8
integral_value = simpson_13_rule(f, a, b, n)
print("Estimated value of the integral using Simpson's 1/3 rule:", integral_value)
```

### 2. Write Python program to obtain the approximate real root of 

\[
x^4 - 8x^2 - 4 = 0
\]

using Regula-Falsi method.

```python
# Function for the equation
def g(x):
    return x**4 - 8*x**2 - 4

# Regula-Falsi method implementation
def regula_falsi(func, a, b, tol=1e-5, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("Function has the same signs at the ends of the interval.")
        
    for _ in range(max_iter):
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        if abs(func(c)) < tol:
            return c
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2  # Return the midpoint if max_iter is reached

# Parameters
root = regula_falsi(g, -3, 3)  # Example interval [-3, 3]
print("Approximate real root using Regula-Falsi method:", root)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to estimate the value of the integral 

\[
\int_0^1 x^5 \, dx
\]

using Trapezoidal rule (n=10).

```python
# Trapezoidal rule implementation
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        integral += f(a + i * h)
        
    return integral * h

# Define the function
def f(x):
    return x**5

# Estimate the integral from 0 to 1 with n = 10
a, b, n = 0, 1, 10
integral_value = trapezoidal_rule(f, a, b, n)
print("Estimated value of the integral using Trapezoidal rule:", integral_value)
```

### 2. Write Python program to find sin(35)° using Newton backward interpolation formula for the data:

\[
\begin{align*}
\text{sin}(30^\circ) & = 0.5 \\
\text{sin}(35^\circ) & = 0.5736 \\
\text{sin}(40^\circ) & = 0.6428 \\
\text{sin}(45^\circ) & = 0.7071
\end{align*}
\]

```python
# Backward interpolation function
def backward_interpolation(x, x_values, y_values):
    n = len(y_values)
    h = x_values[1] - x_values[0]
    k = int((x - x_values[0]) / h)
    
    # Compute the difference table
    diff_table = np.zeros((n, n))
    for i in range(n):
        diff_table[i][0] = y_values[i]
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i][j - 1] - diff_table[i + 1][j - 1]
    
    # Calculate the interpolated value
    result = y_values[k]
    factor = 1
    for j in range(1, k + 1):
        factor *= (x - x_values[k - j]) / h
        result += (factor * diff_table[k - j][j]) / np
