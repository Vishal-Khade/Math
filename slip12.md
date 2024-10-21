
# Slip No. 12

## Q.1. Attempt any two of the following. [10]

### 1. Using Python, evaluate each of the following expressions.
#### (a) \( 23 \mod 2 + 9 - (3 + 7) \times 10 \div 2 \)

```python
# Evaluating the expression
result_a = 23 % 2 + 9 - (3 + 7) * 10 / 2
print("Result of a:", result_a)
```

#### (b) \( 35 \times 10 \text{ floor division } 3 + 15 \mod 3 \)

```python
# Evaluating the expression
result_b = 35 * 10 // 3 + 15 % 3
print("Result of b:", result_b)
```

#### (c) \( 3^5 - 25 + 4 \text{ floor division } 7 \)

```python
# Evaluating the expression
result_c = 3**5 - 25 + 4 // 7
print("Result of c:", result_c)
```

### 2. Use a while loop in Python to find odd positive integers between 25 to 50.

```python
# Finding odd positive integers between 25 and 50
number = 25
odd_numbers = []

while number <= 50:
    if number % 2 != 0:
        odd_numbers.append(number)
    number += 1

print("Odd positive integers between 25 and 50:", odd_numbers)
```

### 3. For matrix \( A = \begin{bmatrix} 1 & 0 & 5 \\ 2 & 1 & 6 \\ 3 & 4 & 0 \\ 4 & -1 & 2 \end{bmatrix} \), apply the following operations.

```python
import numpy as np

# Define matrix A
A = np.array([[1, 0, 5],
              [2, 1, 6],
              [3, 4, 0],
              [4, -1, 2]])

# a. Delete 2nd row
A_deleted_row = np.delete(A, 1, axis=0)  # Deleting 2nd row (index 1)
print("Matrix A after deleting 2nd row:\n", A_deleted_row)

# b. Delete 1st column
A_deleted_column = np.delete(A, 0, axis=1)  # Deleting 1st column (index 0)
print("Matrix A after deleting 1st column:\n", A_deleted_column)

# c. Add column [9, 9] as 2nd column
new_column = np.array([[9], [9], [9], [9]])
A_with_new_column = np.insert(A, 1, new_column, axis=1)  # Inserting as 2nd column (index 1)
print("Matrix A after adding new column as 2nd column:\n", A_with_new_column)
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python code to find the eigenvalues and corresponding eigenvectors of the matrix 
\[ \begin{bmatrix} 1 & 3 & 3 \\ 2 & 2 & 3 \\ 4 & 2 & 1 \end{bmatrix} \].

```python
# Define the matrix
matrix = np.array([[1, 3, 3],
                   [2, 2, 3],
                   [4, 2, 1]])

# Finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### 2. Write a Python program to find the product of n natural numbers using a while loop.

```python
# Finding the product of n natural numbers
n = 5  # Change n to find product of different natural numbers
product = 1
i = 1

while i <= n:
    product *= i
    i += 1

print("Product of first", n, "natural numbers:", product)
```

### 3. Generate all prime numbers between 1 to 200 using Python code.

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Generating prime numbers between 1 and 200
primes = [num for num in range(1, 201) if is_prime(num)]
print("Prime numbers between 1 and 200:", primes)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral 
\[ R_{0}^{\pi} \sin(x) \, dx \] 
using Simpson’s \( \frac{1}{3} \) rule (n=5).

```python
def simpson_13(a, b, n, f):
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

# Function to integrate
def f(x):
    return np.sin(x)

a = 0
b = np.pi
n = 5  # Must be even
integral_value = simpson_13(a, b, n, f)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to diagonalize the matrix 
\[ \begin{bmatrix} 3 & -2 \\ 6 & -4 \end{bmatrix} \]
and find matrices \( P \) and \( D \).

```python
# Define the matrix
matrix = np.array([[3, -2],
                   [6, -4]])

# Finding eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Diagonal matrix D
D = np.diag(eigenvalues)

# Matrix P (eigenvectors)
P = eigenvectors

print("Matrix P (eigenvectors):\n", P)
print("Diagonal Matrix D (eigenvalues):\n", D)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of 
\[ x^3 - 2x - 5 = 0 \] 
in the interval [2, 3] using the Regula-Falsi method.

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
    return x**3 - 2*x - 5

a = 2
b = 3
n = 10
root = regula_falsi(a, b, f, n)
print("Approximate real root:", root)
```

### 2. Write a Python program to estimate the value of the integral 
\[ R_{1}^{5} \frac{1}{(1+x)} \, dx \] 
using the Trapezoidal rule (n=10).

```python
def trapezoidal(a, b, n, f):
    h = (b - a) / n
    I = (f(a) + f(b)) / 2

    for i in range(1, n):
        k = a + i * h
        I += f(k)

    I *= h
    return I

# Function to integrate
def f(x):
    return 1 / (1 + x)

a = 1
b = 5
n = 10
integral_value = trapezoidal(a, b, n, f)

print("Estimated value of the integral:", integral_value)
```

