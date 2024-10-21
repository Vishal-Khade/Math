
# Slip No. 20

## Q.1. Attempt any two of the following. [10]

### 1. Write Python code to print first n natural numbers and their square roots of input integer n.

```python
import math

# Input from user
n = int(input("Enter a positive integer n: "))

print(f"First {n} natural numbers and their square roots:")
for i in range(1, n + 1):
    print(f"Natural number: {i}, Square root: {math.sqrt(i)}")
```

### 2. Use Python code to find the sum of squares of the first twenty-five natural numbers.

```python
# Calculate the sum of squares of the first 25 natural numbers
sum_of_squares = sum(i**2 for i in range(1, 26))
print("Sum of squares of the first 25 natural numbers:", sum_of_squares)
```

### 3. Write Python code to find all positive divisors of a given number n.

```python
# Function to find all positive divisors of a number
def find_divisors(n):
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    return divisors

# Input from user
n = int(input("Enter a positive integer to find its divisors: "))
divisors = find_divisors(n)
print(f"Positive divisors of {n}: {divisors}")
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python code to display the tuple ‘I am Indian’ and the second letter in this tuple.

```python
# Create a tuple
my_tuple = ('I', ' ', 'a', 'm', ' ', 'I', 'n', 'd', 'i', 'a', 'n')

# Display the tuple
print("Tuple:", my_tuple)

# Display the second letter
second_letter = my_tuple[1]
print("Second letter in the tuple:", second_letter)
```

### 2. Write Python code to display the matrix whose all entries are 10 and order is (4,6).

```python
import numpy as np

# Create a matrix of order (4,6) with all entries as 10
matrix = np.full((4, 6), 10)
print("Matrix with all entries as 10:\n", matrix)
```

### 3. Write Python program to diagonalize the matrix 

\[
A = \begin{pmatrix} 3 & -2 \\ 6 & -4 \end{pmatrix}
\]

and find matrix P and D.

```python
from scipy.linalg import eigh

# Define the matrix A
A = np.array([[3, -2],
              [6, -4]])

# Diagonalization
eigenvalues, eigenvectors = eigh(A)

# P is the matrix of eigenvectors and D is the diagonal matrix of eigenvalues
P = eigenvectors
D = np.diag(eigenvalues)

print("Matrix P (Eigenvectors):\n", P)
print("Matrix D (Eigenvalues):\n", D)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral 

\[
R_1^3 \cos(x) \, dx
\]

using Simpson’s \( \left(\frac{3}{8}\right) \) rule (n=6).

```python
import numpy as np

def f(x):
    return np.cos(x)

def simpsons_38_rule(a, b, n):
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3.")

    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1, n):
        k = a + i * h
        if i % 3 == 0:
            integral += 2 * f(k)
        else:
            integral += 3 * f(k)

    integral *= 3 * h / 8
    return integral

# Parameters
a = 1
b = 3
n = 6
integral_value = simpsons_38_rule(a, b, n)

print("Estimated value of the integral R_1^3 cos(x) dx:", integral_value)
```

### 2. Write Python program to evaluate interpolate value \( f(5) \) of the given data.

\[
\begin{align*}
x: & \quad 1 \quad 2 \quad 3 \quad 6 \\
Y=f(x): & \quad 2 \quad 6 \quad 12 \quad 42
\end{align*}
\]

```python
# Data points
x_values = np.array([1, 2, 3, 6])
y_values = np.array([2, 6, 12, 42])

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

# Evaluate f(5)
interpolated_value = lagrange_interpolation(5, x_values, y_values)
print("Interpolated value f(5):", interpolated_value)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of 

\[
x^3 - 5x - 9 = 0
\]

in \([2,3]\) using the Regula-Falsi method.

```python
# Function to find the root using Regula-Falsi method
def f(x):
    return x**3 - 5*x - 9

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

### 2. Write Python program to estimate the value of the integral 

\[
R_1^5 (x^3 - 3x + 2) \, dx
\]

using the Trapezoidal rule (n=5).

```python
def g(x):
    return x**3 - 3*x + 2

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    integral = (g(a) + g(b)) / 2

    for i in range(1, n):
        k = a + i * h
        integral += g(k)

    integral *= h
    return integral

# Parameters
a = 1
b = 5
n = 5
integral_value = trapezoidal_rule(a, b, n)

print("Estimated value of the integral R_1^5 (x^3 - 3x + 2) dx:", integral_value)
```
```
