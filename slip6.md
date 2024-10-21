
# Slip No. 6

## Q.1. Attempt any two of the following. [10]

### 1. Using Python, evaluate each of the following expressions:

a. \( 23 \mod 2 + 9 - (3 + 7) \times 10 \div 2 \)

```python
result_a = 23 % 2 + 9 - (3 + 7) * 10 / 2
print("Result of expression a:", result_a)
```

b. \( 35 \times 10 \text{ floor division } 3 + 15 \mod 3 \)

```python
result_b = 35 * 10 // 3 + 15 % 3
print("Result of expression b:", result_b)
```

c. \( 3^5 - 2^5 + 4 \text{ floor division } 7 \)

```python
result_c = 3**5 - 2**5 + 4 // 7
print("Result of expression c:", result_c)
```

### 2. Write Python code to list names and roll numbers of 5 students in B.Sc. (Computer Science).

```python
students = [
    {"name": "Alice", "roll_number": 1},
    {"name": "Bob", "roll_number": 2},
    {"name": "Charlie", "roll_number": 3},
    {"name": "David", "roll_number": 4},
    {"name": "Eve", "roll_number": 5}
]

print("Students in B.Sc. (Computer Science):")
for student in students:
    print(f"Name: {student['name']}, Roll Number: {student['roll_number']}")
```

### 3. Write Python code to find the maximum and minimum elements in the given list.

```python
numbers = [7, 8, 71, 32, 49, -5, 7, 7, 0, 1, 6]

max_value = max(numbers)
min_value = min(numbers)

print("Maximum Element:", max_value)
print("Minimum Element:", min_value)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, construct an identity matrix of order 10 and hence find the determinant, trace, and transpose of it.

```python
import numpy as np

# Constructing identity matrix of order 10
identity_matrix = np.eye(10)

# Finding determinant, trace, and transpose
determinant = np.linalg.det(identity_matrix)
trace = np.trace(identity_matrix)
transpose = identity_matrix.T

print("Identity Matrix (10x10):\n", identity_matrix)
print("Determinant:", determinant)
print("Trace:", trace)
print("Transpose of Identity Matrix:\n", transpose)
```

### 2. Write Python code to find the value of the function \( f(x, y) = x^2 - 2xy + 4 \) at the points (2, 0) and (1, -1).

```python
def f(x, y):
    return x**2 - 2*x*y + 4

value1 = f(2, 0)
value2 = f(1, -1)

print("f(2, 0):", value1)
print("f(1, -1):", value2)
```

### 3. Find numbers between 1 to 200 that are divisible by 7 using Python code.

```python
divisible_by_7 = [num for num in range(1, 201) if num % 7 == 0]
print("Numbers between 1 to 200 divisible by 7:", divisible_by_7)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral \( \int_0^\pi (x - \sin(x))dx \) using Simpson’s \( \frac{1}{3} \) rule (n=5).

```python
import numpy as np

def simpson_13(a, b, n, f):
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
def integrand(x):
    return x - np.sin(x)

a = 0
b = np.pi
n = 5
integral_value = simpson_13(a, b, n, integrand)

print("Estimated value of the integral:", integral_value)
```

### 2. Write Python code to diagonalize matrix \( A = \begin{pmatrix} 3 & -2 \\ 6 & -4 \end{pmatrix} \) and find matrix \( P \) with diagonalized \( A \) and diagonal matrix \( D \).

```python
from scipy.linalg import eigh

A = np.array([[3, -2],
              [6, -4]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(A)

# Diagonal matrix D
D = np.diag(eigenvalues)

# Matrix P
P = eigenvectors

print("Diagonal Matrix D:\n", D)
print("Matrix P:\n", P)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of \( x^3 - 2x - 5 = 0 \) in [2, 3] using the Regula-Falsi method.

```python
def regula_falsi(a, b, f, n):
    for i in range(n):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        print(f'Iteration {i+1}, c = {c}, f(c) = {f(c)}')
    return c

def f(x):
    return x**3 - 2*x - 5

a = 2
b = 3
n = 10
root = regula_falsi(a, b, f, n)
print("Approximate root:", root)
```

### 2. Write a Python program to estimate the value of the integral \( \int_1^5 \frac{1}{1+x}dx \) using the Trapezoidal rule (n=10).

```python
def trapezoidal(a, b, n, f):
    h = (b - a) / n
    I = (f(a) + f(b)) / 2
    
    for i in range(1, n):
        I += f(a + i * h)
    
    I *= h
    return I

def integrand(x):
    return 1 / (1 + x)

a = 1
b = 5
n = 10
integral_value = trapezoidal(a, b, n, integrand)
print("Estimated value of the integral:", integral_value)
```

