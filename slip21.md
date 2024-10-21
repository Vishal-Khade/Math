
# Slip No. 21

## Q.1. Attempt any two of the following. [10]

### 1. Write Python code to display multiplication tables of numbers 20 to 30.

```python
# Display multiplication tables from 20 to 30
for i in range(20, 31):
    print(f"\nMultiplication table of {i}:")
    for j in range(1, 11):
        print(f"{i} x {j} = {i * j}")
```

### 2. Write Python code to list name and birth date of 5 students in your class.

```python
# List of students with their birth dates
students = [
    {"name": "Alice", "birth_date": "2003-01-15"},
    {"name": "Bob", "birth_date": "2002-02-22"},
    {"name": "Charlie", "birth_date": "2004-03-10"},
    {"name": "David", "birth_date": "2003-04-05"},
    {"name": "Eve", "birth_date": "2001-12-12"}
]

print("Name and Birth Date of Students:")
for student in students:
    print(f"Name: {student['name']}, Birth Date: {student['birth_date']}")
```

### 3. Write Python function \( f(a, b) = (4a + b, 3(a - 6b)) \), find the value of \( f(12, 25) \).

```python
# Define the function f(a, b)
def f(a, b):
    return (4*a + b, 3*(a - 6*b))

# Calculate the value for f(12, 25)
result = f(12, 25)
print("Value of f(12, 25):", result)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python, construct the following matrices.

#### a. Matrix of order 5×6 with all entries 1.

```python
import numpy as np

# Matrix of order 5x6 with all entries 1
matrix_ones = np.ones((5, 6))
print("Matrix of order 5x6 with all entries 1:\n", matrix_ones)
```

#### b. Zero matrix of order 27 × 33.

```python
# Zero matrix of order 27x33
zero_matrix = np.zeros((27, 33))
print("Zero matrix of order 27x33:\n", zero_matrix)
```

#### c. Identity matrix of order 5.

```python
# Identity matrix of order 5
identity_matrix = np.eye(5)
print("Identity matrix of order 5:\n", identity_matrix)
```

### 2. Write Python code to perform the \( R_2 + 2R_1 \) row operation on the given matrix.

\[
R = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3 \end{pmatrix}
\]

```python
# Given matrix
R = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]])

# Perform row operation R2 + 2R1
R[1] = R[1] + 2 * R[0]

print("Matrix after row operation R2 + 2R1:\n", R)
```

### 3. Write Python code to find all the eigenvalues and the eigenvectors of the matrix.

\[
A = \begin{pmatrix} 2 & -1 & -1 & 0 \\ -1 & 3 & -1 & -1 \\ -1 & -1 & 3 & -1 \\ -1 & -1 & -1 & 2 \end{pmatrix}
\]

```python
# Define the matrix A
A = np.array([[2, -1, -1, 0],
              [-1, 3, -1, -1],
              [-1, -1, 3, -1],
              [-1, -1, -1, 2]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to find the approximate root of the equation 

\[
x^5 + 3x + 1 = 0
\]

by using the Newton-Raphson method.

```python
def f(x):
    return x**5 + 3*x + 1

def f_prime(x):
    return 5*x**4 + 3

def newton_raphson(x0, tol=1e-5, max_iter=100):
    for _ in range(max_iter):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return x0

# Initial guess
root = newton_raphson(-1)  # A reasonable starting point
print("Approximate root using Newton-Raphson method:", root)
```

### 2. Write a Python program to evaluate interpolate value \( f(3) \) of the given data.

\[
\begin{align*}
x: & \quad 1 \quad 2 \quad 3 \quad 4 \\
Y=f(x): & \quad 11 \quad 22 \quad 33 \quad 66
\end{align*}
\]

```python
# Data points
x_values = np.array([1, 2, 3, 4])
y_values = np.array([11, 22, 33, 66])

# Lagrange interpolation function
def lagrange_interpolation(x, x_values, y_values):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

# Evaluate f(3)
interpolated_value = lagrange_interpolation(3, x_values, y_values)
print("Interpolated value f(3):", interpolated_value)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of 

\[
x \sin(x) + \cos(x) = 0
\]

by using the Regula-Falsi method.

```python
import numpy as np

# Function for the equation
def g(x):
    return x * np.sin(x) + np.cos(x)

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
root = regula_falsi(g, 0, 1)  # Example interval [0, 1]
print("Approximate real root using Regula-Falsi method:", root)
```

### 2. Write Python program to find \(\sin(37^\circ)\) using the Newton backward interpolation formula for the data:

\[
\begin{align*}
\sin(30^\circ) & = 0.5 \\
\sin(35^\circ) & = 0.5736 \\
\sin(40^\circ) & = 0.6428 \\
\sin(45^\circ) & = 0.7071
\end{align*}
\]

```python
# Data points for interpolation
x_values = np.array([30, 35, 40, 45])
y_values = np.array([0.5, 0.5736, 0.6428, 0.7071])

# Newton backward interpolation
def newton_backward_interpolation(x, x_values, y_values):
    n = len(x_values)
    h = x_values[1] - x_values[0]
    k = (x - x_values[-1]) / h
    
    # Calculate the backward differences
    b_diff = np.zeros((n, n))
    b_diff[:, 0] = y_values

    for j in range(1, n):
        for i in range(n - j):
            b_diff[i][j] = b_diff[i + 1][j - 1] - b_diff[i][j - 1]

    result = b_diff[n-1][0]
    for j in range(1, n):
        prod = 1
        for i in range(j):
            prod *= (k + i)
        result += (prod * b_diff[n - j - 1][j]) / np.math.factorial(j)
    
    return result

# Evaluate sin(37) using Newton backward interpolation
sin_37 = newton_backward_interpolation(37, x_values, y_values)
print("Estimated value of sin(37 degrees):", sin_37)
```
```
