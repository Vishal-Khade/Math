
# Slip No. 17

## Q.1. Attempt any two of the following. [10]

### 1. Write the Python code to print ‘Python is bad’ and ‘Python is wonderful’, where wonderful is a global variable and bad is a local variable.

```python
# Global variable
wonderful = "Python is wonderful"

def print_statements():
    # Local variable
    bad = "Python is bad"
    print(bad)
    print(wonderful)

# Call the function
print_statements()
```

### 2. Write Python code to evaluate eigenvalues and eigenvectors of the following matrix:

\[
A = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}
\]

```python
import numpy as np

# Define the matrix A
A = np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 1]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### 3. Write Python code to find \(a\), \(b\), and \(c\) such that \(a^2 + b^2 = c^2\) where \(1 \leq a, b, c \leq 50\).

```python
# Find values of a, b, and c
found = False

for a in range(1, 51):
    for b in range(a, 51):  # Start b from a to avoid duplicates
        c_squared = a**2 + b**2
        c = int(c_squared**0.5)
        if c_squared == c**2 and 1 <= c <= 50:
            print(f"a = {a}, b = {b}, c = {c}")
            found = True

if not found:
    print("No such a, b, c found in the range.")
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code construct any two matrices A and B to show that \((AB)^{-1} = B^{-1}A^{-1}\).

```python
# Define matrices A and B
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Calculate inverses
A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# Calculate the product and its inverse
AB = np.dot(A, B)
AB_inv = np.linalg.inv(AB)

# Show that (AB)^-1 = B^-1 A^-1
left_side = AB_inv
right_side = np.dot(B_inv, A_inv)

print("Left Side (AB)^-1:\n", left_side)
print("Right Side B^-1 A^-1:\n", right_side)

# Check if they are close enough
print("Are they equal?", np.allclose(left_side, right_side))
```

### 2. Use `linsolve` code in Python to solve the following system of linear equations:

\[
\begin{align*}
x - 2y + 3z &= 7 \\
2x + y + z &= 4 \\
-3x + 2y - 2z &= -10
\end{align*}
\]

```python
from sympy import symbols, Eq, linsolve

# Define the symbols
x, y, z = symbols('x y z')

# Define the equations
eq1 = Eq(x - 2*y + 3*z, 7)
eq2 = Eq(2*x + y + z, 4)
eq3 = Eq(-3*x + 2*y - 2*z, -10)

# Solve the system of equations
solution = linsolve([eq1, eq2, eq3], x, y, z)
print("Solution of the system of equations:", solution)
```

### 3. Write Python code to find the trace and transpose of the matrix:

\[
A = \begin{pmatrix} 1 & 3 & 3 \\ 2 & 2 & 3 \\ 4 & 2 & 1 \end{pmatrix}
\]

```python
# Define the matrix A
A = np.array([[1, 3, 3],
              [2, 2, 3],
              [4, 2, 1]])

# Calculate trace and transpose
trace_A = np.trace(A)
transpose_A = A.T

print("Trace of A:", trace_A)
print("Transpose of A:\n", transpose_A)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to find \( f(3) \) of the functional value \( f(1)=2, f(2)=10, f(4)=68 \) by using Lagrange method.

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
x_values = [1, 2, 4]
y_values = [2, 10, 68]

# Target value
x_target = 3
f_3 = lagrange_interpolation(x_values, y_values, x_target)

print(f"f(3) = {f_3}")
```

### 2. Write Python program to estimate a root of an equation \( x^5 - 5x + 6 = 0 \) using Newton–Raphson method correct up to four decimal places.

```python
# Function definition
def f(x):
    return x**5 - 5*x + 6

def df(x):
    return 5*x**4 - 5  # Derivative of f(x)

# Newton-Raphson method
def newton_raphson(initial_guess, tolerance=1e-4, max_iterations=100):
    x = initial_guess
    for _ in range(max_iterations):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tolerance:
            return round(x_new, 4)  # Round to four decimal places
        x = x_new
    return None  # No convergence

# Starting guess
initial_guess = 1  # Starting point
root = newton_raphson(initial_guess)
print("Estimated root using Newton-Raphson method:", root)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of \( x^2 - 2x - 1 = 0 \) by using Regula-Falsi method in the interval [1, 3].

```python
def regula_falsi(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have different signs.")
    
    for _ in range(max_iter):
        c = b - f(b) * (a - b) / (f(a) - f(b))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return None  # No convergence

# Function for f(x)
def f(x):
    return x**2 - 2*x - 1

# Interval [1, 3]
root_regula_falsi = regula_falsi(f, 1, 3)
print("Approximate real root using Regula-Falsi method:", root_regula_falsi)
```

### 2. Write Python program to estimate the value of the integral 

\[
R_{0}^{1} x^2 \, dx 
\]

using the Trapezoidal rule (n=10).

```python
def f(x):
    return x**2

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    I = (f(a) + f(b)) / 2

    for i in range(1, n):
        k = a + i * h
        I += f(k)

    I *= h
    return I

# Parameters
a = 0
b = 1
n = 10
integral_value = trapezoidal_rule(a, b, n)

print("Estimated value of the integral R_0^1 x^2 dx:", integral_value)
```
```

