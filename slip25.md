
# Slip No. 25

## Q.1. Attempt any two of the following. [10]

### 1. Write Python program to print the integers between 1 and 1000 which are multiples of 7.

```python
# Print integers between 1 and 1000 that are multiples of 7
multiples_of_7 = [i for i in range(1, 1001) if i % 7 == 0]
print("Integers between 1 and 1000 which are multiples of 7:", multiples_of_7)
```

### 2. Write Python program to print whether the given number is divisible by 3 or 5 or 7.

```python
# Check if a given number is divisible by 3, 5, or 7
number = int(input("Enter a number: "))
if number % 3 == 0 or number % 5 == 0 or number % 7 == 0:
    print(f"{number} is divisible by 3, 5, or 7.")
else:
    print(f"{number} is not divisible by 3, 5, or 7.")
```

### 3. Write Python code to find \( A + B \) and \( B \cdot A \) for the given matrices.

```python
import numpy as np

# Define matrices A and B
A = np.array([[4, 2, 4],
              [4, -1, 1],
              [2, 4, 2]])

B = np.array([[5, 2, 3],
              [3, -7, 5],
              [3, 1, -1]])

# Calculate A + B and B * A
sum_matrix = A + B
product_matrix = np.dot(B, A)

print("Matrix A + B:\n", sum_matrix)
print("Matrix B * A:\n", product_matrix)
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python program to find the area and circumference of a circle with radius r.

```python
import math

# Calculate the area and circumference of a circle
def circle_properties(radius):
    area = math.pi * radius**2
    circumference = 2 * math.pi * radius
    return area, circumference

# Input radius
r = float(input("Enter the radius of the circle: "))
area, circumference = circle_properties(r)
print(f"Area: {area}, Circumference: {circumference}")
```

### 2. Use Python code to solve the following system of equations by Gauss elimination method.

```python
import numpy as np

# Coefficient matrix
A = np.array([[1, 1, 2],
              [-1, -2, 3],
              [3, -7, 6]])

# Right-hand side
b = np.array([7, 6, 1])

# Solve using numpy
solution = np.linalg.solve(A, b)
print("Solution of the system of equations (x, y, z):", solution)
```

### 3. Write Python code to find eigenvalues, eigenvectors of the matrix and determine whether the matrix is diagonalizable.

```python
# Define the matrix
matrix = np.array([[1, -1, 1],
                   [-1, 1, -1],
                   [1, -1, 1]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Check diagonalizability
is_diagonalizable = len(set(eigenvalues)) == len(eigenvalues)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("Is the matrix diagonalizable?", is_diagonalizable)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to find the approximate root of the equation \( f(x) = x^2 - 50 \) by using Newton Raphson method.

```python
# Function for Newton Raphson method
def f(x):
    return x**2 - 50

def df(x):
    return 2*x  # Derivative of f(x)

def newton_raphson(x0, tol=1e-5, max_iter=100):
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return x0

# Initial guess
x0 = 7
root = newton_raphson(x0)
print("Approximate root using Newton Raphson method:", root)
```

### 2. Write Python program to evaluate \( f(2.4) \) by forward difference formula of the given data.

```python
# Forward Difference Formula Implementation
def forward_difference(x, y, x_value):
    n = len(y)
    h = x[1] - x[0]
    
    # Calculate forward differences
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y.copy()
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    
    # Calculate the forward difference value
    k = (x_value - x[0]) / h
    result = diff_table[0][0]
    
    for j in range(1, n):
        term = diff_table[0][j]
        for m in range(j):
            term *= (k - m)
        result += term
    
    return result

# Data
x = np.array([0, 1, 2, 3])
y = np.array([11, 10, 11, 21])

# Evaluate f(2.4)
x_value = 2.4
f_value = forward_difference(x, y, x_value)
print("Estimated value of f(2.4) using forward difference formula:", f_value)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to estimate the value of the integral 

\[
\int_0^1 \sin^2(\pi x) \, dx
\]

using Simpson’s (1/3) rule (n=6).

```python
# Simpson's 1/3 Rule Implementation
def simpson_13_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even.")
    
    h = (b - a) / n
    integral = f(a) + f(b)
    
    for i in range(1, n, 2):
        integral += 4 * f(a + i * h)
    for i in range(2, n, 2):
        integral += 2 * f(a + i * h)
        
    return integral * h / 3

# Define the function
def f(x):
    return np.sin(np.pi * x)**2

# Estimate the integral from 0 to 1 with n = 6
a, b, n = 0, 1, 6
integral_value = simpson_13_rule(f, a, b, n)
print("Estimated value of the integral using Simpson's 1/3 rule:", integral_value)
```

### 2. Write Python program to find \( f(4) \) using Lagrange’s interpolation formula from the data: \( f(1) = 6, f(2) = 9, f(5) = 30, f(7) = 54 \).

```python
# Lagrange Interpolation Implementation
def lagrange_interpolation(x_values, y_values, x):
    total_sum = 0
    n = len(x_values)
    
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        total_sum += term
    
    return total_sum

# Data
x_values = np.array([1, 2, 5, 7])
y_values = np.array([6, 9, 30, 54])

# Evaluate f(4)
x_to_interpolate = 4
f_value = lagrange_interpolation(x_values, y_values, x_to_interpolate)
print("Estimated value f(4) using Lagrange's interpolation formula:", f_value)
```

