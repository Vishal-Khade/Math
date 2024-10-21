Here's the solution for Slip No. 5 formatted in markdown, suitable for GitHub:

```markdown
# Slip No. 5

## Q.1. Attempt any two of the following. [10]

### 1. Using the sympy module of Python, find the following for the matrices

\[
A =
\begin{pmatrix}
-1 & 1 & 0 \\
8 & 5 & 2 \\
2 & -6 & 2
\end{pmatrix}
\]

\[
B =
\begin{pmatrix}
9 & 0 & 3 \\
1 & 4 & 1 \\
1 & 0 & -1
\end{pmatrix}
\]

#### (a) \(2A + B\)

```python
import sympy as sp

# Define the matrices A and B
A = sp.Matrix([[-1, 1, 0], [8, 5, 2], [2, -6, 2]])
B = sp.Matrix([[9, 0, 3], [1, 4, 1], [1, 0, -1]])

# Calculate 2A + B
result_a = 2 * A + B
print("2A + B:\n", result_a)
```

#### (b) \(3A - 5B\)

```python
# Calculate 3A - 5B
result_b = 3 * A - 5 * B
print("3A - 5B:\n", result_b)
```

#### (c) \(A^{-1}\)

```python
# Calculate the inverse of A
result_c = A.inv()
print("A^{-1}:\n", result_c)
```

#### (d) \(B^3\)

```python
# Calculate B^3
result_d = B**3
print("B^3:\n", result_d)
```

#### (e) \(A^T + B^T\)

```python
# Calculate A^T + B^T
result_e = A.T + B.T
print("A^T + B^T:\n", result_e)
```

### 2. Evaluate the following expressions in Python.

#### (a) M = [1, 2, 3, 4], Find length M.

```python
M = [1, 2, 3, 4]
length_M = len(M)
print("Length of M:", length_M)
```

#### (b) L = "XYZ" + "pqr", Find L.

```python
L = "XYZ" + "pqr"
print("L:", L)
```

#### (c) s = 'Make In India', Find (s[:7]) & (s[:9]).

```python
s = 'Make In India'
substring_7 = s[:7]
substring_9 = s[:9]
print("s[:7]:", substring_7)
print("s[:9]:", substring_9)
```

### 3. Use Python code to generate the square root of numbers from 21 to 49.

```python
import math

square_roots = {i: math.sqrt(i) for i in range(21, 50)}
print("Square roots from 21 to 49:", square_roots)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python, construct the following matrices.

1. An identity matrix of order 10 × 10.
2. A zero matrix of order 7 × 3.
3. A ones matrix of order 5 × 4.

```python
import numpy as np

# 1. Identity matrix 10x10
identity_matrix = np.eye(10)
print("Identity Matrix 10x10:\n", identity_matrix)

# 2. Zero matrix 7x3
zero_matrix = np.zeros((7, 3))
print("Zero Matrix 7x3:\n", zero_matrix)

# 3. Ones matrix 5x4
ones_matrix = np.ones((5, 4))
print("Ones Matrix 5x4:\n", ones_matrix)
```

### 2. Using `linsolve` command in Python, solve the following system of linear equations:

\[
x - 2y + 3z = 7 \\
2x + y + z = 4 \\
-3x + 2y - 2z = -10
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

### 3. Generate all relatively prime numbers to 111 that are less than 150 using Python code.

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Find all numbers less than 150 that are relatively prime to 111
relatively_prime_numbers = [i for i in range(1, 150) if gcd(i, 111) == 1]
print("Relatively prime numbers to 111 less than 150:", relatively_prime_numbers)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python code to find eigenvalues and corresponding eigenvectors of the matrix

\[
A =
\begin{pmatrix}
1 & 3 & 3 \\
2 & 2 & 3 \\
4 & 2 & 1
\end{pmatrix}
\]

and hence find matrix P that diagonalizes A.

```python
# Define the matrix A
A = np.array([[1, 3, 3],
              [2, 2, 3],
              [4, 2, 1]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Construct the diagonal matrix D
D = np.diag(eigenvalues)

# Verify the diagonalization
P = eigenvectors
print("Matrix P that diagonalizes A:\n", P)
print("D = P^-1 * A * P:\n", np.linalg.inv(P) @ A @ P)
```

### 2. Write a Python program to estimate a root of an equation \(f(x) = 3x^2 + 4x - 10\) using the Newton-Raphson method correct up to four decimal places.

```python
def f(x):
    return 3 * x**2 + 4 * x - 10

def g(x):
    return 6 * x + 4

def N_R(x0, n, f, g):
    for i in range(1, n):
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        print("\nIteration %d, x1=%0.6f and f(x1)=%0.6f" % (i, x1, f(x1)))
    print('Root by Newton Raphson Method x1=', round(x1, 4))

# Initial guess
initial_guess = 1
iterations = 10

N_R(initial_guess, iterations, f, g)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of \(x^3 - 4x - 9 = 0\) by using the Regula-Falsi method.

```python
def f(x):
    return x**3 - 4*x - 9

def R_F(a, b, n, f):
    for i in range(1, n):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        print('\nIteration %d, c=%0.6f and f(c)=%0.6f' % (i, c, f(c)))
    print('\nRoot by Regula Falsi method c=', round(c, 4))

# Initial guesses
a = 2
b = 3
iterations = 10

R_F(a, b, iterations, f)
```

### 2. Write a Python program to evaluate \(f(3.5)\) by the forward difference formula of the given data.

```
x = [1, 2, 3, 4, 5]
Y = [41, 62, 65, 50, 17]
```

```python
import numpy as np

def forward_difference(Y):
    n = len(Y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = Y  # First column is the function values

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table



x = [1, 2, 3, 4, 5]
Y = [41, 62, 65, 50, 17]

# Calculate the forward difference table
diff_table = forward_difference(Y)
print("Forward Difference Table:\n", diff_table)

# Now we can evaluate f(3.5) using the first forward difference
h = x[1] - x[0]  # Assuming uniform spacing
x0 = 3  # the value to estimate
n = 1  # order of forward difference

# Calculate the value of f(3.5)
value = diff_table[2][0] + ((x0 - x[2]) / h) * diff_table[2][1]
print("f(3.5) using forward difference formula:", value)
```

This format makes the code easy to read, understand, and execute. You can run this in a Python environment that supports the libraries used (`numpy`, `sympy`).
```
