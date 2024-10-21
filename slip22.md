
# Slip No. 22

## Q.1. Attempt any two of the following. [10]

### 1. Write Python code to sort a tuple in ascending order (49, 17, 23, 54, 36, 72).

```python
# Sorting a tuple in ascending order
numbers = (49, 17, 23, 54, 36, 72)
sorted_numbers = tuple(sorted(numbers))
print("Sorted tuple:", sorted_numbers)
```

### 2. Find the values of the following expressions if x and y are true and z is false.
#### (a) \( (x \lor y) \land z \)
#### (b) \( (x \land y) \lor \neg z \)
#### (c) \( (x \land \neg y) \lor (x \land z) \)

```python
# Define boolean variables
x = True
y = True
z = False

# Evaluate the expressions
expr_a = (x or y) and z
expr_b = (x and y) or not z
expr_c = (x and not y) or (x and z)

print("Value of (x or y) and z:", expr_a)
print("Value of (x and y) or not z:", expr_b)
print("Value of (x and not y) or (x and z):", expr_c)
```

### 3. Write Python code to find the tuple ‘MATHEMATICS’ from range 3 to 9.

```python
# Creating the tuple 'MATHEMATICS'
word = "MATHEMATICS"
result_tuple = tuple(word[3:9])
print("Tuple from range 3 to 9:", result_tuple)
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python program that prints whether the given number is positive, negative, or zero.

```python
# Check if a number is positive, negative or zero
number = float(input("Enter a number: "))

if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")
```

### 2. Write Python program to find the sum of first n natural numbers.

```python
# Find the sum of first n natural numbers
n = int(input("Enter a natural number n: "))
sum_n = n * (n + 1) // 2
print("Sum of first", n, "natural numbers is:", sum_n)
```

### 3. Using Python accept the matrix 

\[
A = \begin{pmatrix} 1 & -3 & 2 & -4 \\ -3 & 9 & -1 & 5 \\ 5 & -2 & 6 & -3 \\ -4 & 12 & 2 & 7 \end{pmatrix}
\]

Find the Null space, Column space and rank of the matrix.

```python
from sympy import Matrix

# Define the matrix A
A = Matrix([[1, -3, 2, -4],
             [-3, 9, -1, 5],
             [5, -2, 6, -3],
             [-4, 12, 2, 7]])

# Calculate Null space
null_space = A.nullspace()
print("Null space of the matrix A:", null_space)

# Calculate Column space
column_space = A.columnspace()
print("Column space of the matrix A:", column_space)

# Calculate rank of the matrix
rank = A.rank()
print("Rank of the matrix A:", rank)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write Python program to find the approximate root of 

\[
f(x) = x^3 - 10x^2 + 5 = 0
\]

using the Newton-Raphson method. Take \( x_0 = 0.5 \).

```python
# Define the function f(x) and its derivative
def f(x):
    return x**3 - 10*x**2 + 5

def f_prime(x):
    return 3*x**2 - 20*x

def newton_raphson(x0, tol=1e-5, max_iter=100):
    for _ in range(max_iter):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return x0

# Initial guess
root = newton_raphson(0.5)  # Starting point
print("Approximate root using Newton-Raphson method:", root)
```

### 2. Write Python program to evaluate the interpolate value \( f(2) \) of the given data.

\[
\begin{align*}
x: & \quad 11 \quad 12 \quad 13 \quad 14 \\
Y=f(x): & \quad 21 \quad 19 \quad 27 \quad 64
\end{align*}
\]

```python
# Data points for interpolation
x_values = np.array([11, 12, 13, 14])
y_values = np.array([21, 19, 27, 64])

# Lagrange interpolation function
def lagrange_interpolation(x, x_values, y_values):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

# Evaluate f(2)
interpolated_value = lagrange_interpolation(2, x_values, y_values)
print("Interpolated value f(2):", interpolated_value)
```

## Q.3. b. Attempt any one of the following. [8]

### 1. Write Python program to obtain the approximate real root of 

\[
x^3 - x^2 - 2 = 0
\]

in \([1,2]\), using the Regula-Falsi method.

```python
# Function for the equation
def g(x):
    return x**3 - x**2 - 2

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
root = regula_falsi(g, 1, 2)  # Example interval [1, 2]
print("Approximate real root using Regula-Falsi method:", root)
```

### 2. Using Python accept the matrix 

\[
A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 5 & 3 \\ 1 & 0 & 8 \end{pmatrix}
\]

Find the transpose of the matrix, determinant, inverse of the matrix. Also, reduce the matrix to reduced row echelon form and diagonalize it.

```python
# Define the matrix A
A = Matrix([[1, 2, 3],
             [2, 5, 3],
             [1, 0, 8]])

# Calculate the transpose of the matrix
transpose_A = A.transpose()
print("Transpose of matrix A:\n", transpose_A)

# Calculate the determinant of the matrix
determinant_A = A.det()
print("Determinant of matrix A:", determinant_A)

# Calculate the inverse of the matrix
inverse_A = A.inv()
print("Inverse of matrix A:\n", inverse_A)

# Reduce the matrix to reduced row echelon form
rref_A, _ = A.rref()
print("Reduced row echelon form of matrix A:\n", rref_A)
```
```
