
# Slip No. 15

## Q.1. Attempt any two of the following. [10]

### 1. Using for loop on Python, find range from 1 to 11 integers.

```python
# Using for loop to find range from 1 to 11
for i in range(1, 12):
    print(i)
```

### 2. Use Python code to find:

#### (a) sin(75)

```python
import math

# Calculate sin(75)
sin_75 = math.sin(math.radians(75))
print("sin(75) =", sin_75)
```

#### (b) pi/2

```python
# Calculate pi/2
pi_over_2 = math.pi / 2
print("pi/2 =", pi_over_2)
```

#### (c) e

```python
# Calculate e
e_value = math.e
print("e =", e_value)
```

#### (d) cos(56)

```python
# Calculate cos(56)
cos_56 = math.cos(math.radians(56))
print("cos(56) =", cos_56)
```

### 3. Write Python program to find diameter, area, circumference of the circle with radius 5.

```python
# Circle radius
radius = 5

# Calculate diameter, area, and circumference
diameter = 2 * radius
area = math.pi * (radius ** 2)
circumference = 2 * math.pi * radius

print("Diameter of the circle:", diameter)
print("Area of the circle:", area)
print("Circumference of the circle:", circumference)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, construct any three matrices \( A, B, \) and \( C \) to show that \( (A+B)+C = A+(B+C) \).

```python
import numpy as np

# Define matrices A, B, and C
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])

# Calculate (A + B) + C
left_side = (A + B) + C
print("Left Side ((A + B) + C):\n", left_side)

# Calculate A + (B + C)
right_side = A + (B + C)
print("Right Side (A + (B + C)):\n", right_side)

# Check if they are equal
assert np.array_equal(left_side, right_side), "The two expressions are not equal!"
```

### 2. Using Python, find the eigenvalues and corresponding eigenvectors of the matrix 

\[ \begin{pmatrix} 3 & -2 \\ 6 & -4 \end{pmatrix} \]

```python
# Define the matrix
matrix = np.array([[3, -2], [6, -4]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### 3. Generate all prime numbers between 1000 to 2000 using Python program.

```python
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Generate prime numbers between 1000 and 2000
primes = [num for num in range(1000, 2001) if is_prime(num)]
print("Prime numbers between 1000 and 2000:", primes)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral 

\[ R_{0}^{6} e^x \, dx \] 

using Simpson’s \( \frac{1}{3} \) rule (n=6).

```python
def f(x):
    return np.exp(x)

def simpson_13(a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even.")
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
a = 0
b = 6
n = 6  # Must be even
integral_value = simpson_13(a, b, n)

print("Estimated value of the integral:", integral_value)
```

### 2. Write Python program to estimate a root of an equation 

\[ f(x) = 3x - \cos(x) - 1 \]

using Newton–Raphson method correct up to four decimal places.

```python
# Function definition
def f(x):
    return 3*x - np.cos(x) - 1

def df(x):
    return 3 + np.sin(x)  # Derivative of f(x)

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
initial_guess = 1.0
root = newton_raphson(initial_guess)
print("Estimated root using Newton-Raphson method:", root)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of 

\[ x^3 - 4x - 9 = 0 \] 

by using Regula-Falsi method.

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
    return x**3 - 4*x - 9

# Interval [2, 3]
root_regula_falsi = regula_falsi(f, 2, 3)
print("Approximate real root using Regula-Falsi method:", root_regula_falsi)
```

### 2. Write Python program to evaluate interpolate value \( f(2.2) \) of the given data:

\[ f(2) = 0.593, f(2.5) = 0.816, f(3) = 1.078 \]

```python
# Given data
x_values = [2, 2.5, 3]
f_values = [0.593, 0.816, 1.078]

def linear_interpolation(x, x0, x1, f0, f1):
    return f0 + (x - x0) * (f1 - f0) / (x1 - x0)

# Evaluate f(2.2) using linear interpolation
x_target = 2.2
if x_target < x_values[0] or x_target > x_values[-1]:
    raise ValueError("x_target is out of the interpolation range.")

# Determine the interval for interpolation
if x_target <= 2.5:
    f_interpolated = linear_interpolation(x_target, x_values[0], x_values[1], f_values[0], f_values[1])
else:
    f_interpolated = linear_interpolation(x_target, x_values[1], x_values[2], f_values[1], f_values[2])

print("Interpolated value f(2.2):", f_interpolated)
```
