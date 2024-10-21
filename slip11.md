
# Slip No. 11

## Q.1. Attempt any two of the following. [10]

### 1. Evaluate the following expressions in Python.
#### (a) Given \( M = [1,2,3,4,5,6,7] \), find the length of \( M \).

```python
# List M
M = [1, 2, 3, 4, 5, 6, 7]

# Finding the length of M
length_M = len(M)
print("Length of M:", length_M)
```

#### (b) Given \( L = "XY" + "pqr" \), find \( L \).

```python
# Concatenating strings
L = "XY" + "pqr"
print("Value of L:", L)
```

#### (c) Given \( s = 'Make In India' \), find \( (s[:5]) \) & \( (s[:9]) \).

```python
# Given string s
s = 'Make In India'

# Slicing the string
s1 = s[:5]
s2 = s[:9]
print("s[:5]:", s1)
print("s[:9]:", s2)
```

### 2. Use Python to evaluate expressions for the following matrix:
#### \( A = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix} \)

#### (a) Eigenvalues of \( A \).

```python
import numpy as np

# Define matrix A
A = np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 1]])

# Finding eigenvalues
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of A:", eigenvalues)
```

#### (b) Determinant of \( A \).

```python
# Finding determinant
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)
```

#### (c) Inverse of \( A \).

```python
# Finding inverse of A
try:
    inv_A = np.linalg.inv(A)
    print("Inverse of A:\n", inv_A)
except np.linalg.LinAlgError:
    print("Matrix A is singular and cannot be inverted.")
```

### 3. Write Python code to reverse the string \( S = [3,4,5,6,7,8,9,10,11,12,13] \).

```python
# Given list S
S = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Reversing the list
reversed_S = S[::-1]
print("Reversed S:", reversed_S)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code to list the names of 5 teachers in your college with their subjects.

```python
# List of teachers and their subjects
teachers = [
    {"name": "Mr. A", "subject": "Mathematics"},
    {"name": "Ms. B", "subject": "Physics"},
    {"name": "Dr. C", "subject": "Chemistry"},
    {"name": "Mr. D", "subject": "Biology"},
    {"name": "Ms. E", "subject": "Computer Science"},
]

for teacher in teachers:
    print(f"Name: {teacher['name']}, Subject: {teacher['subject']}")
```

### 2. Use `linsolve` command in Python to solve the following system of linear equations.
\[
\begin{align*}
x - 2y + 3z & = 7 \\
2x + y + z & = 4 \\
-3x + 2y - 2z & = -10
\end{align*}
\]

```python
from sympy import symbols, Eq, linsolve

# Define the variables
x, y, z = symbols('x y z')

# Define the equations
eq1 = Eq(x - 2*y + 3*z, 7)
eq2 = Eq(2*x + y + z, 4)
eq3 = Eq(-3*x + 2*y - 2*z, -10)

# Solve the system of equations
solution = linsolve([eq1, eq2, eq3], (x, y, z))
print("Solution to the system of equations:", solution)
```

### 3. Generate all the prime numbers between 51 to 100 using a Python program.

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Generating prime numbers between 51 and 100
primes = [num for num in range(51, 101) if is_prime(num)]
print("Prime numbers between 51 and 100:", primes)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral 
\[ R_{0}^{10} e^x \, dx \] 
using Simpson’s \( \frac{3}{8} \) rule (Take \( h = 0.5 \)).

```python
def simpson_38(a, b, n, f):
    if n % 3 != 0:
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

# Function to integrate
def f(x):
    return np.exp(x)

a = 0
b = 10
h = 0.5
n = int((b - a) / h)
integral_value = simpson_38(a, b, n, f)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to find the approximate root of the function 
\[ x^5 + 3x + 1 \] 
in the interval [-2, 0] using the Newton-Raphson Method correct up to 4 decimal places.

```python
def f(x):
    return x**5 + 3*x + 1

def f_prime(x):
    return 5*x**4 + 3

# Newton-Raphson method
def newton_raphson(x0, tol=1e-4, max_iter=100):
    for i in range(max_iter):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < tol:
            return round(x1, 4)
        x0 = x1
    return None

x0 = -1  # Initial guess
root = newton_raphson(x0)
print("Approximate root:", root)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of 
\[ x^3 - 4x - 9 = 0 \] 
by using the Regula-Falsi method.

```python
def regula_falsi(a, b, f, n):
    for i in range(n):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        print(f"Iteration {i+1}, c = {c}, f(c) = {f(c)}")
    return c

def f(x):
    return x**3 - 4*x - 9

a = 2
b = 3
n = 10
root = regula_falsi(a, b, f, n)
print("Approximate real root:", root)
```

### 2. Write a Python program to evaluate the interpolated value \( f(153) \) of the given data.
\[
\begin{align*}
x & : 150, 152, 154, 155 \\
Y & : f(x) : 12.247, 12.329, 12.410, 12.490
\end{align*}
\]

```python
# Given data points
x_points = [150, 152, 154, 155]
y_points = [12.247, 12.329, 12.410, 12.490]

# Function to perform linear interpolation
def linear_interpolation(x_points, y_points, x_value):
    return np.interp(x_value, x_points, y_points)

# Estimate f(153)
interpolated_value = linear_interpolation(x_points, y_points, 153)
print("Interpolated value f(153):", interpolated_value)
```

