
# Slip No. 2

## Q.1. Attempt any two of the following. [10]

### 1. Calculate the volume of a sphere with radius r = 7 (V = (4/3)πr³).

```python
import math

r = 7
volume = (4/3) * math.pi * r**3
print("Volume of the sphere:", volume)
```

### 2. String operation ‘+’ on two strings.

#### a. string1 = "Hello", string2 = "World!"
```python
string1 = "Hello"
string2 = "World!"
combined_string = string1 + " " + string2
print("Combined String:", combined_string)
```

#### b. string1 = "Good", string2 = "Morning"
```python
string1 = "Good"
string2 = "Morning"
combined_string = string1 + " " + string2
print("Combined String:", combined_string)
```

### 3. Generate the square of numbers from 20 to 30.
```python
squares = [i**2 for i in range(20, 31)]
print("Squares from 20 to 30:", squares)
```

## Q.2. Attempt any two of the following. [10]

### 1. Find the value of f(-2), f(0), f(2) where f(x) = x² - 5x + 6.
```python
def f(x):
    return x**2 - 5*x + 6

values = [-2, 0, 2]
results = {x: f(x) for x in values}
print("f(-2):", results[-2], ", f(0):", results[0], ", f(2):", results[2])
```

### 2. Find the 10th term of the sequence of function f(x) = x³ + 5x.
```python
def f(x):
    return x**3 + 5*x

# 10th term
term_10 = f(10)
print("10th term of the sequence f(x) = x^3 + 5x is:", term_10)
```

### 3. Using SymPy, find the eigenvalues and corresponding eigenvectors of the matrix A:
\[
A = \begin{bmatrix}
4 & 2 & 2 \\
2 & 4 & 2 \\
2 & 2 & 4
\end{bmatrix}
\]
```python
import numpy as np
import sympy as sp

A = np.array([[4, 2, 2],
              [2, 4, 2],
              [2, 2, 4]])

# Create a sympy matrix
sym_A = sp.Matrix(A)

# Find eigenvalues and eigenvectors
eigenvalues = sym_A.eigenvals()
eigenvectors = sym_A.eigenvects()

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Estimate the value of the integral \( \int_0^1 \frac{1}{1+x^2} dx \) using Simpson’s (1/3) rule (n=4).
```python
import numpy as np

def simpsons_13(a, b, n, f):
    h = (b - a) / n
    I = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            I += 2 * f(x)
        else:
            I += 4 * f(x)
    return (h / 3) * I

def f(x):
    return 1 / (1 + x**2)

# Interval [0, 1], n=4
result = simpsons_13(0, 1, 4, f)
print("Estimated value of integral using Simpson’s 1/3 rule:", result)
```

### 2. Find a real root of \( f(x) = x^3 - 8x - 4 = 0 \) using Newton-Raphson method.
```python
def f(x):
    return x**3 - 8*x - 4

def g(x):
    return 3*x**2 - 8  # Derivative of f

def N_R(x0, n, f, g):
    for i in range(1, n):
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        print("\nIteration %d, x1 = %0.6f and f(x1) = %0.6f" % (i, x1, f(x1)))
    print('Root by Newton Raphson Method: x1 =', x1)

# Initial guess and iterations
N_R(3, 10, f, g)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Approximate real roots of \( x^3 - 2x - 5 = 0 \) in [2, 3] using Regula-Falsi method.
```python
def R_F(a, b, n, f):
    for i in range(1, n):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        print('\nIteration %d, c = %0.6f and f(c) = %0.6f' % (i, c, f(c)))
    print('\nRoot by Regula Falsi method: c =', c)

def f(x):
    return x**3 - 2*x - 5

# Example usage
R_F(2, 3, 10, f)
```

### 2. Evaluate interpolated value \( f(3.5) \) of the given data using Lagrange's method.
```python
def lagrange_interpolation(x_values, y_values, x_to_find):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term = term * (x_to_find - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

x_values = [0, 1, 2, 5]
y_values = [2, 3, 12, 147]

# Interpolating f(3.5)
f_3_5 = lagrange_interpolation(x_values, y_values, 3.5)
print("Interpolated value f(3.5):", f_3_5)
```
```
