
# Slip No. 16

## Q.1. Attempt any two of the following. [10]

### 1. Write Python program to find the absolute value of a given real number (n).

```python
# Function to find absolute value
def absolute_value(n):
    return abs(n)

# Example usage
n = -42.5  # Change this value to test with other numbers
print("Absolute value of", n, "is", absolute_value(n))
```

### 2. Using Python program

Given `List1 = [5, 10, 15, 20, 25, 30]` and `List2 = [7, 14, 21, 28, 35, 42]`, evaluate:

#### (a) `List1 + List2`

```python
# Lists
List1 = [5, 10, 15, 20, 25, 30]
List2 = [7, 14, 21, 28, 35, 42]

# Concatenating lists
result_list_concat = List1 + List2
print("List1 + List2 =", result_list_concat)
```

#### (b) `7 * List1`

```python
# Multiplying List1 by 7
result_list_multiply_7 = 7 * List1
print("7 * List1 =", result_list_multiply_7)
```

#### (c) `11 * List2`

```python
# Multiplying List2 by 11
result_list_multiply_11 = 11 * List2
print("11 * List2 =", result_list_multiply_11)
```

### 3. Write Python program to find the area and circumference of a circle (r=5).

```python
import math

# Circle radius
r = 5

# Calculate area and circumference
area = math.pi * (r ** 2)
circumference = 2 * math.pi * r

print("Area of the circle with radius", r, "is", area)
print("Circumference of the circle with radius", r, "is", circumference)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, find the percentage of marks in five subjects out of 100 each.

```python
# Marks in five subjects
marks = [70, 80, 55, 78, 65]
total_marks = 100 * len(marks)  # Total marks for five subjects

# Calculate percentage
percentage = sum(marks) / total_marks * 100
print("Percentage of marks:", percentage)
```

### 2. Using `sympy` module of Python, find the following terms of vector \( x = [1, -5, 0] \) and \( y = [2, 3, -1] \).

```python
import sympy as sp

# Define vectors
x = sp.Matrix([1, -5, 0])
y = sp.Matrix([2, 3, -1])

# Calculate required terms
term_5x = 5 * x
term_x_plus_y = x + y
term_x_minus_3y = x - 3 * y

print("5x =", term_5x)
print("x + y =", term_x_plus_y)
print("x - 3y =", term_x_minus_3y)
```

### 3. Write Python code to find the determinant and inverse of matrices

\[ A = \begin{pmatrix} 1 & 0 & 5 \\ 2 & 1 & 6 \\ 3 & 4 & 0 \end{pmatrix} \] 

and 

\[ B = \begin{pmatrix} 2 & 5 \\ -1 & 4 \end{pmatrix} \]

```python
# Define matrices A and B
A = np.array([[1, 0, 5], [2, 1, 6], [3, 4, 0]])
B = np.array([[2, 5], [-1, 4]])

# Calculate determinant and inverse of A
det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)

# Calculate determinant and inverse of B
det_B = np.linalg.det(B)
inv_B = np.linalg.inv(B)

print("Determinant of A:", det_A)
print("Inverse of A:\n", inv_A)
print("Determinant of B:", det_B)
print("Inverse of B:\n", inv_B)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to estimate the value of the integral 

\[ R_{0}^{\pi} \sin(x) \, dx \]

using Simpson’s \( \frac{1}{3} \) rule (n=6).

```python
def f(x):
    return np.sin(x)

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
b = math.pi
n = 6  # Must be even
integral_value = simpson_13(a, b, n)

print("Estimated value of the integral R_0^π sin(x) dx:", integral_value)
```

### 2. Write Python program to estimate a root of an equation 

\[ f(x) = x^5 + 5x + 1 \]

using Newton-Raphson method in the interval [-1,0].

```python
# Function definition
def f(x):
    return x**5 + 5*x + 1

def df(x):
    return 5*x**4 + 5  # Derivative of f(x)

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
initial_guess = -0.5  # Within the interval [-1, 0]
root = newton_raphson(initial_guess)
print("Estimated root using Newton-Raphson method:", root)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of 

\[ x^2 - 2x - 1 = 0 \]

by using Regula-Falsi method in the interval [2,3].

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

# Interval [2, 3]
root_regula_falsi = regula_falsi(f, 2, 3)
print("Approximate real root using Regula-Falsi method:", root_regula_falsi)
```

### 2. Write Python program to estimate the value of the integral 

\[ R_{2}^{10} \frac{1}{1+x} \, dx \]

using Trapezoidal rule (n=8).

```python
def f(x):
    return 1 / (1 + x)

def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    I = (f(a) + f(b)) / 2

    for i in range(1, n):
        k = a + i * h
        I += f(k)

    I *= h
    return I

# Parameters
a = 2
b = 10
n = 8
integral_value = trapezoidal_rule(a, b, n)

print("Estimated value of the integral R_2^10 (1+x)^-1 dx:", integral_value)
```
```

