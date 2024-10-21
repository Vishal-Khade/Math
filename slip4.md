
# Slip No. 4

## Q.1. Attempt any two of the following. [10]

### 1. Using Python code, sort the tuple in ascending and descending order: (5, -3, 0, 1, 6, -6, 2).

```python
# Define the tuple
numbers = (5, -3, 0, 1, 6, -6, 2)

# Sort in ascending order
sorted_ascending = tuple(sorted(numbers))
print("Sorted in ascending order:", sorted_ascending)

# Sort in descending order
sorted_descending = tuple(sorted(numbers, reverse=True))
print("Sorted in descending order:", sorted_descending)
```

### 2. Write a Python program which deals with concatenation and repetition of lists.

```python
# Define the lists
List1 = [15, 20, 25, 30, 35, 40]
List2 = [7, 14, 21, 28, 35, 42]

# (a) Concatenation of List1 and List2
concatenated_list = List1 + List2
print("Concatenated List1 and List2:", concatenated_list)

# (b) Repetition of List1
repeated_list1 = 9 * List1
print("List1 repeated 9 times:", repeated_list1)

# (c) Repetition of List2
repeated_list2 = 7 * List2
print("List2 repeated 7 times:", repeated_list2)
```

### 3. Write Python code to find the square of odd numbers from 1 to 20 using a while loop.

```python
# Initialize variables
number = 1
squares_of_odds = []

# Using while loop to find squares of odd numbers
while number <= 20:
    if number % 2 != 0:  # Check if the number is odd
        squares_of_odds.append(number ** 2)
    number += 1

print("Squares of odd numbers from 1 to 20:", squares_of_odds)
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

### 2. Find the data type of the following data by using Python code.

```python
# Data to check
data_list = [5, 31.25, 8 + 4j, "Mathematics", 49]

# Checking data types
data_types = {str(data): type(data) for data in data_list}
print("Data types of the given data:\n", data_types)
```

### 3. Write a Python program to find the determinant of matrices A and B.

\[
A =
\begin{pmatrix}
1 & 0 & 5 \\
2 & 1 & 6 \\
3 & 4 & 0
\end{pmatrix}
\]

\[
B =
\begin{pmatrix}
2 & 5 \\
-1 & 4
\end{pmatrix}
\]

```python
import numpy as np

# Define matrices A and B
A = np.array([[1, 0, 5],
              [2, 1, 6],
              [3, 4, 0]])

B = np.array([[2, 5],
              [-1, 4]])

# Calculate determinants
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)

print("Determinant of matrix A:", det_A)
print("Determinant of matrix B:", det_B)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral \(\int_0^{\pi} x \sin(x) \, dx\) using Simpson’s \(1/3\) rule (n=6).

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
    return x * np.sin(x)

# Interval [0, pi], n=6
result = simpsons_13(0, np.pi, 6, f)
print("Estimated value of integral using Simpson’s 1/3 rule:", result)
```

### 2. Write a Python program to estimate a root of an equation \(f(x) = 3x - \cos(x) - 1\) using the Newton-Raphson method correct up to four decimal places.

```python
import numpy as np

def f(x):
    return 3*x - np.cos(x) - 1

def f_prime(x):
    return 3 + np.sin(x)

def newton_raphson(x0, tolerance, max_iterations):
    for i in range(max_iterations):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < tolerance:
            return x1
        x0 = x1
    return None

# Initial guess
initial_guess = 0.5
tolerance = 1e-4
max_iterations = 100

root = newton_raphson(initial_guess, tolerance, max_iterations)
print("Estimated root of the equation:", round(root, 4))
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to find all positive prime numbers less than a given number \(n\).

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def find_primes_less_than_n(n):
    primes = [i for i in range(2, n) if is_prime(i)]
    return primes

# Example usage
n = 20
primes = find_primes_less_than_n(n)
print(f"Prime numbers less than {n}:", primes)
```

### 2. Write a Python program to evaluate \(f(2.5)\) by the forward difference formula of the given data.

```
x = [0, 1, 2, 3]
Y = [2, 1, 2, 10]
```

```python
import numpy as np

def forward_difference(Y):
    n = len(Y)
    # Create a table to store forward differences
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = Y  # First column is the function values

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table

def interpolate(x_values, y_values, x_to_find):
    n = len(x_values)
    diff_table = forward_difference(y_values)
    result = y_values[0]

    h = x_values[1] - x_values[0]  # Assuming uniform spacing
    for i in range(1, n):
        term = diff_table[0][i]
        for j in range(i):
            term *= (x_to_find - x_values[j]) / (h * (i - j))
        result += term

    return result

x_values = [0, 1, 2, 3]
y_values = [2, 1, 2, 10]

# Interpolating f(2.5)
f_2_5 = interpolate(x_values, y_values, 2.5)
print("Evaluated value f(2.5):", f_2_5)
```
```
