# Slip No. 3

## Q.1. Attempt any two of the following. [10]

### 1. Write Python code to test whether a given number is divisible by 2, 3, or 5.

```python
def check_divisibility(num):
    if num % 2 == 0:
        print(f"{num} is divisible by 2.")
    if num % 3 == 0:
        print(f"{num} is divisible by 3.")
    if num % 5 == 0:
        print(f"{num} is divisible by 5.")
    if num % 2 != 0 and num % 3 != 0 and num % 5 != 0:
        print(f"{num} is not divisible by 2, 3, or 5.")

# Example usage
check_divisibility(30)
```

### 2. Repeat the following string 11 times using the string operator ‘*’ in Python.

```python
# Repeating strings
string_a = "LATEX" * 11
string_b = "MATLAB" * 11
print(string_a)
print(string_b)
```

### 3. Use Python code to find the sum of the first thirty natural numbers.

```python
# Sum of first 30 natural numbers
sum_natural_numbers = sum(range(1, 31))
print("Sum of the first thirty natural numbers:", sum_natural_numbers)
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

### 2. Using Python, find the eigenvalues and corresponding eigenvectors of the matrix:

\[
\begin{pmatrix}
3 & -2 \\
6 & -4
\end{pmatrix}
\]

```python
import numpy as np

# Define the matrix
A = np.array([[3, -2], [6, -4]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### 3. Generate all the prime numbers between 1 to 100 using Python code.

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

primes = [i for i in range(1, 101) if is_prime(i)]
print("Prime numbers between 1 and 100:", primes)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral \(\int_0^{\pi} \sin(x) \, dx\) using Simpson’s \(1/3\) rule (n=6).

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
    return np.sin(x)

# Interval [0, pi], n=6
result = simpsons_13(0, np.pi, 6, f)
print("Estimated value of integral using Simpson’s 1/3 rule:", result)
```

### 2. Write a Python program to evaluate the third-order forward difference of the given data.

```
x = [0, 1, 2, 3]
Y = [1, 0, 1, 10]
```

```python
def forward_difference(Y):
    n = len(Y)
    # Create a table to store forward differences
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = Y  # First column is the function values

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table

Y = [1, 0, 1, 10]
diff_table = forward_difference(Y)
print("Forward difference table:\n", diff_table)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to evaluate \(f(3.5)\) of the given data.

```
x = [1, 2, 3, 4, 5]
Y = [30, 50, 55, 40, 11]
```

```python
def lagrange_interpolation(x_values, y_values, x_to_find):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x_to_find - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

x_values = [1, 2, 3, 4, 5]
y_values = [30, 50, 55, 40, 11]

# Interpolating f(3.5)
f_3_5 = lagrange_interpolation(x_values, y_values, 3.5)
print("Interpolated value f(3.5):", f_3_5)
```

### 2. Write a Python program to estimate the value of the integral \(\int_1^2 (1+x) \, dx\) using the Trapezoidal rule (n=5).

```python
import numpy as np

def trapezoidal_rule(a, b, n, f):
    h = (b - a) / n
    I = (f(a) + f(b)) / 2
    for i in range(1, n):
        I += f(a + i * h)
    return I * h

def f(x):
    return 1 + x

# Interval [1, 2], n=5
result = trapezoidal_rule(1, 2, 5, f)
print("Estimated value of integral using Trapezoidal rule:", result)
```
```
