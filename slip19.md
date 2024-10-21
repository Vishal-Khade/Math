
# Slip No. 19

## Q.1. Attempt any two of the following. [10]

### 1. Write Python code to display multiplication tables of numbers 2 to 10.

```python
# Multiplication tables from 2 to 10
for i in range(2, 11):
    print(f"Multiplication table for {i}:")
    for j in range(1, 11):
        print(f"{i} x {j} = {i * j}")
    print()  # Blank line for better readability
```

### 2. Write Python code to check if a number is Zero, Odd or Even.

```python
# Function to check if a number is Zero, Odd or Even
def check_number(num):
    if num == 0:
        return "Zero"
    elif num % 2 == 0:
        return "Even"
    else:
        return "Odd"

# Test the function
number = int(input("Enter a number: "))
result = check_number(number)
print(f"The number {number} is {result}.")
```

### 3. Write Python code to list names and birth dates of 5 students in your class.

```python
# List of students and their birth dates
students = [
    {"name": "Alice", "birth_date": "2001-05-14"},
    {"name": "Bob", "birth_date": "2002-08-19"},
    {"name": "Charlie", "birth_date": "2000-11-25"},
    {"name": "Diana", "birth_date": "2003-01-12"},
    {"name": "Ethan", "birth_date": "2001-03-05"}
]

# Display the names and birth dates
print("Name\t\tBirth Date")
print("-------------------------")
for student in students:
    print(f"{student['name']}\t{student['birth_date']}")
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python code to find the transpose and inverse of matrix 

\[
A = \begin{pmatrix} 1 & 2 & 2 \\ 2 & 1 & 2 \\ 2 & 2 & 1 \end{pmatrix}
\]

```python
import numpy as np

# Define the matrix A
A = np.array([[1, 2, 2],
              [2, 1, 2],
              [2, 2, 1]])

# Transpose of A
transpose_A = A.T

# Inverse of A
inverse_A = np.linalg.inv(A)

print("Transpose of A:\n", transpose_A)
print("Inverse of A:\n", inverse_A)
```

### 2. Declare the matrix 

\[
A = \begin{pmatrix} 5 & 2 & 5 & 4 \\ 10 & 3 & 4 & 6 \\ 2 & 0 & -1 & 11 \end{pmatrix}
\]

find the row echelon form and the rank of matrix A.

```python
from sympy import Matrix

# Define the matrix A
A = Matrix([[5, 2, 5, 4],
             [10, 3, 4, 6],
             [2, 0, -1, 11]])

# Row echelon form
row_echelon_form = A.rref()[0]

# Rank of the matrix
rank_A = A.rank()

print("Row Echelon Form of A:\n", row_echelon_form)
print("Rank of matrix A:", rank_A)
```

### 3. Declare the matrix 

\[
A = \begin{pmatrix} 2 & -1 & 2 & 7 \\ 4 & 7 & 3 & 4 \\ 4 & 2 & 0 & -1 \end{pmatrix}
\]

find the matrices L and U such that \( A = LU \).

```python
# Define the matrix A
A = np.array([[2, -1, 2, 7],
              [4, 7, 3, 4],
              [4, 2, 0, -1]])

# LU decomposition using scipy
from scipy.linalg import lu

P, L, U = lu(A)

print("Lower Triangular Matrix L:\n", L)
print("Upper Triangular Matrix U:\n", U)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral 

\[
R_0^1 \frac{1}{1+x^2} \, dx
\]

by using Simpson’s \( \left(\frac{3}{8}\right) \) rule (n=6).

```python
def f(x):
    return 1 / (1 + x**2)

def simpsons_38_rule(a, b, n):
    if n % 3 != 0:  # n must be a multiple of 3
        raise ValueError("n must be a multiple of 3.")
        
    h = (b - a) / n
    I = f(a) + f(b)

    for i in range(1, n):
        k = a + i * h
        if i % 3 == 0:
            I += 2 * f(k)
        else:
            I += 3 * f(k)

    I *= 3 * h / 8
    return I

# Parameters
a = 0
b = 1
n = 6
integral_value = simpsons_38_rule(a, b, n)

print("Estimated value of the integral R_0^1 (1/(1+x^2)) dx:", integral_value)
```

### 2. Write Python program to obtain the approximate real root of 

\[
x^3 - 8x - 4 = 0
\]

using Regula-Falsi method.

```python
def f(x):
    return x**3 - 8*x - 4

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
root = regula_falsi(f, 2, 3)
print("Approximate real root using Regula-Falsi method:", root)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to estimate the value of the integral 

\[
R_0^1 x^2 \, dx
\]

using Trapezoidal rule (n=5).

```python
def g(x):
    return x**2

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
integral_value_x2 = trapezoidal_rule(a, b, n)

print("Estimated value of the integral R_0^1 (x^2) dx:", integral_value_x2)
```

### 2. Write Python program to find \(\sin(42^\circ)\) using Newton backward interpolation formula for the data:

\[
\begin{align*}
\sin(30^\circ) & = 0.5, \\
\sin(35^\circ) & = 0.5736, \\
\sin(40^\circ) & = 0.6428, \\
\sin(45^\circ) & = 0.7071
\end{align*}
\]

```python
# Backward interpolation function
def backward_interpolation(x, y, x_target):
    n = len(x)
    h = x[1] - x[0]
    m = int((x_target - x[0]) / h)
    
    # Create a difference table
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    # Interpolation
    result = y[0]
    p = 1

    for j in range(1, m + 1):
        p *= (x_target - x[j - 1]) / (j * h)
        result += p * diff_table[0][j]

    return result

# Known points
x_values = [30, 35, 40, 45]
y_values = [0.5, 0.5736, 0.6428, 0.7071]

