
# Slip No. 14

## Q.1. Attempt any two of the following. [10]

### 1. Use print code on Python with \( a = 4, b = 6, c = 8, d = 12 \).

```python
# Assign values to variables
a = 4
b = 6
c = 8
d = 12

# Evaluate and print expressions
print("a + c =", a + c)             # (a)
print("a * b =", a * b)             # (b)
print("c ** d =", c ** d)           # (c)
print("a / b =", a / b)             # (d)

# Evaluate and print the expression
expression_result = 3 + (9 - 2) / 7 * 2 ** 2
print("Expression result =", expression_result)  # (e)
```

### 2. For the following two statements, use the ‘+’ string operation on Python.

```python
# Concatenate strings
string1_a = "Hello"
string2_a = "World!"
result_a = string1_a + ", " + string2_a
print("Concatenated String A:", result_a)

string1_b = "Good"
string2_b = "Morning"
result_b = string1_b + ", " + string2_b
print("Concatenated String B:", result_b)
```

### 3. Use Python loop to print ‘Hallo’, i, ‘You Learn Python’ where \( i = ['Saurabh', 'Akash', 'Sandeep', 'Ram', 'Sai'] \).

```python
# List of names
names = ['Saurabh', 'Akash', 'Sandeep', 'Ram', 'Sai']

# Loop to print the desired output
for i in names:
    print(f'Hallo, {i}, You Learn Python')
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, construct any two matrices \( A \) and \( B \) and show that \( A + B = B + A \), and find \( A - B \).

```python
import numpy as np

# Define matrices A and B
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Calculate A + B and B + A
sum_ab = A + B
sum_ba = B + A

print("A + B:\n", sum_ab)
print("B + A:\n", sum_ba)
assert np.array_equal(sum_ab, sum_ba), "A + B should equal B + A"

# Calculate A - B
difference_ab = A - B
print("A - B:\n", difference_ab)
```

### 2. Write Python program to find the sequence of function \( f(x) = x + 5 \), where \( -5 \leq x \leq 5 \).

```python
# Function definition
def f(x):
    return x + 5

# Generate and print the sequence
sequence = [f(x) for x in range(-5, 6)]
print("Sequence of f(x) = x + 5 from -5 to 5:", sequence)
```

### 3. Using sympy module of Python, find the eigenvalues and corresponding eigenvectors of the matrix

\[ A = \begin{pmatrix} 4 & 2 & 2 \\ 2 & 4 & 2 \\ 2 & 2 & 4 \end{pmatrix} \]

```python
import sympy as sp

# Define the matrix A
A = sp.Matrix([[4, 2, 2], [2, 4, 2], [2, 2, 4]])

# Calculate eigenvalues and eigenvectors
eigenvalues = A.eigenvals()
eigenvectors = A.eigenvects()

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
for eig in eigenvectors:
    print(f"Eigenvalue: {eig[0]}, Multiplicity: {eig[1]}, Eigenvector: {eig[2]}")
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral 

\[ R_{0}^{1} \frac{1}{1 + x^2} \, dx \] 

using Simpson’s \( \frac{1}{3} \) rule (n=4).

```python
import numpy as np

def f(x):
    return 1 / (1 + x**2)

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
b = 1
n = 4  # Must be even
integral_value = simpson_13(a, b, n)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to obtain a real root of 

\[ f(x) = x^3 - 8x - 4 = 0 \] 

using Newton–Raphson method.

```python
# Function definition
def f(x):
    return x**3 - 8*x - 4

def df(x):
    return 3*x**2 - 8  # Derivative of f(x)

# Newton-Raphson method
def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    for _ in range(max_iterations):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tolerance:
            return x_new
        x = x_new
    return None  # No convergence

# Starting guess
initial_guess = 3
root = newton_raphson(initial_guess)
print("Real root of f(x) =", root)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of 

\[ x^3 - 2x - 5 = 0 \] 

in [2,3] using Regula-Falsi method.

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
    return x**3 - 2*x - 5

# Interval [2, 3]
root_regula_falsi = regula_falsi(f, 2, 3)
print("Approximate real root using Regula-Falsi method:", root_regula_falsi)
```

### 2. Write a Python program to evaluate approximate value of \( f(1.5) \) by using the forward difference formula of the given data.

```python
# Given data
x_values = [1, 2, 3, 4, 5]
Y = [30, 50, 65, 40, 18]

# Forward difference table
n = len(Y)
forward_diff = np.zeros((n, n))
forward_diff[:, 0] = Y

for j in range(1, n):
    for i in range(n - j):
        forward_diff[i][j] = forward_diff[i + 1][j - 1] - forward_diff[i][j - 1]

# Calculate f(1.5) using forward difference formula
x0 = x_values[0]  # base point
h = x_values[1] - x_values[0]  # step size
x_target = 1.5  # target value

# Calculate the value
f_x_target = forward_diff[0][0]  # f(x0)
num_terms = int((x_target - x0) / h)

for i in range(1, num_terms + 1):
    term = (1)
    for j in range(i):
        term *= (x_target - (x0 + j * h)) / h
    f_x_target += forward_diff[0][i] * term

print("Approximate value of f(1.5):", f_x_target)
```
```

### Instructions for Execution
- Copy each code block and run it in your local Python environment.
- Ensure you have the `numpy` and `sympy` libraries installed using `pip install numpy sympy` if necessary.

