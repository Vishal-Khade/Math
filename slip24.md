
# Slip No. 24

## Q.1. Attempt any two of the following. [10]

### 1. Write Python program to calculate the surface area of sphere \( A = 4\pi r^2 \).

```python
import math

# Calculate the surface area of a sphere
def surface_area_sphere(radius):
    return 4 * math.pi * radius**2

# Input radius
r = float(input("Enter the radius of the sphere: "))
area = surface_area_sphere(r)
print("Surface area of the sphere with radius", r, "is:", area)
```

### 2. Use Python code to find the remainder after dividing by ‘n’ any integers.

```python
# Function to find remainder after dividing by n
def remainder_division(dividend, divisor):
    return dividend % divisor

# Input values
dividend = int(input("Enter the dividend (integer): "))
divisor = int(input("Enter the divisor (n): "))
remainder = remainder_division(dividend, divisor)
print(f"The remainder of {dividend} divided by {divisor} is: {remainder}")
```

### 3. Write Python program to print all integers between 1 to 50 that are divisible by 3 and 7.

```python
# Print all integers between 1 to 50 that are divisible by 3 and 7
divisible_by_3_and_7 = [i for i in range(1, 51) if i % 3 == 0 and i % 7 == 0]
print("Integers between 1 to 50 that are divisible by 3 and 7:", divisible_by_3_and_7)
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python program to find perfect squares between 1 to 100.

```python
# Find perfect squares between 1 to 100
perfect_squares = [i**2 for i in range(1, 11) if i**2 <= 100]
print("Perfect squares between 1 to 100:", perfect_squares)
```

### 2. Write Python program to print whether the given natural number is divisible by 5 and less than 100.

```python
# Check if a number is divisible by 5 and less than 100
number = int(input("Enter a natural number: "))
if number < 100 and number % 5 == 0:
    print(f"{number} is divisible by 5 and less than 100.")
else:
    print(f"{number} is not divisible by 5 or is not less than 100.")
```

### 3. Write Python program to diagonalize the matrix 

\[
\begin{pmatrix}
2 & -3 \\
4 & -6
\end{pmatrix}
\]

and find matrix \( P \) and \( D \).

```python
import numpy as np

# Define the matrix
A = np.array([[2, -3],
              [4, -6]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Diagonal matrix D
D = np.diag(eigenvalues)

# Matrix P (eigenvectors)
P = eigenvectors

print("Matrix P (eigenvectors):\n", P)
print("Diagonal matrix D (eigenvalues):\n", D)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral 

\[
\int_1^3 \cos(x) \, dx
\]

using Simpson’s (3/8) rule (n=5).

```python
# Simpson's 3/8 Rule Implementation
def simpson_38_rule(f, a, b, n):
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3.")
    
    h = (b - a) / n
    integral = f(a) + f(b)
    
    for i in range(1, n, 3):
        integral += 3 * (f(a + i * h) + f(a + (i + 1) * h))
    for i in range(2, n, 3):
        integral += 2 * f(a + i * h)
        
    return integral * 3 * h / 8

# Define the function
def f(x):
    return np.cos(x)

# Estimate the integral from 1 to 3 with n = 5
a, b, n = 1, 3, 5
integral_value = simpson_38_rule(f, a, b, n)
print("Estimated value of the integral using Simpson's 3/8 rule:", integral_value)
```

### 2. Write Python program to evaluate \( f(1.9) \) by using the backward difference formula of the given data.

```python
# Backward Difference Formula Implementation
def backward_difference(x, y, x_value):
    n = len(y)
    h = x[1] - x[0]
    
    # Calculate the backward differences
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    
    # Calculate the backward difference value
    k = (x_value - x[0]) / h
    result = diff_table[n - 1][0]
    
    for j in range(1, n):
        factor = 1
        for m in range(j):
            factor *= (k - m)
        result += (factor * diff_table[n - 1 - j][j]) / np.math.factorial(j)
    
    return result

# Data
x = np.array([1, 2, 3, 4])
y = np.array([11, 10, 15, 10])

# Evaluate f(1.9)
x_value = 1.9
f_value = backward_difference(x, y, x_value)
print("Estimated value of f(1.9) using backward difference formula:", f_value)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of 

\[
x^3 - 5x - 9 = 0
\]

in \([2,4]\) using the Regula-Falsi method.

```python
# Function for the equation
def g(x):
    return x**3 - 5*x - 9

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
root = regula_falsi(g, 2, 4)  # Example interval [2, 4]
print("Approximate real root using Regula-Falsi method:", root)
```

### 2. Write Python program to evaluate the interpolated value \( f(17) \) of the given data.

```python
# Newton's divided difference interpolation implementation
def interpolate(x_values, y_values, x):
    n = len(x_values)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_values.copy()
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x_values[i + j] - x_values[i])
    
    # Calculate the interpolated value
    result = divided_diff[0][0]
    for j in range(1, n):
        term = divided_diff[0][j]
        for k in range(j):
            term *= (x - x_values[k])
        result += term
    
    return result

# Data
x_values = np.array([12, 22, 32, 62])
y_values = np.array([25, 65, 125, 425])

# Evaluate f(17)
x_to_interpolate = 17
f_value = interpolate(x_values, y_values, x_to_interpolate)
print("Interpolated value f(17):", f_value)
```

