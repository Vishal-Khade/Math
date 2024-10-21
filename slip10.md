
# Slip No. 10

## Q.1. Attempt any two of the following. [10]

### 1. Using Python, evaluate each of the following expressions.
#### a. \( 50 \mod 5 + 11 - (13 + 7) \times 10 \div 5 \)

```python
# Evaluating the expression
result_a = 50 % 5 + 11 - (13 + 7) * 10 / 5
print("Result of expression a:", result_a)
```

#### b. \( 60 \times 20 \text{ floor division } 3 + 15 \mod 3 \)

```python
# Evaluating the expression
result_b = 60 * 20 // 3 + 15 % 3
print("Result of expression b:", result_b)
```

#### c. \( 2^7 - 23 + 8 \text{ floor division } 4 \)

```python
# Evaluating the expression
result_c = 2**7 - 23 + 8 // 4
print("Result of expression c:", result_c)
```

### 2. Using Python code
```python
# Given lists
List1 = [5, 10, 15, 20, 25, 30]
List2 = [7, 14, 21, 28, 35, 42]

# Evaluating the expressions
result_list_sum = List1 + List2
result_list1_mult = 3 * List1
result_list2_mult = 5 * List2

print("List1 + List2:", result_list_sum)
print("3 * List1:", result_list1_mult)
print("5 * List2:", result_list2_mult)
```

### 3. Write Python code to find the area of a triangle whose base is 10 and height is 15.

```python
# Given dimensions of the triangle
base = 10
height = 15

# Calculating the area of the triangle
area = 0.5 * base * height
print("Area of the triangle:", area)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python, construct the following matrices.
#### 1. An identity matrix of order 10 × 10.
#### 2. A zero matrix of order 7 × 3.
#### 3. A ones matrix of order 5 × 4.

```python
import numpy as np

# Constructing the matrices
identity_matrix = np.eye(10)
zero_matrix = np.zeros((7, 3))
ones_matrix = np.ones((5, 4))

print("Identity Matrix (10x10):\n", identity_matrix)
print("Zero Matrix (7x3):\n", zero_matrix)
print("Ones Matrix (5x4):\n", ones_matrix)
```

### 2. Write a Python program to find the value of the function \( f(x) = x^2 + x \), for \( -5 \leq x \leq 5 \).

```python
# Define the function
def f(x):
    return x**2 + x

# Calculate f(x) for the range -5 to 5
results = {x: f(x) for x in range(-5, 6)}
print("Function values f(x) for -5 <= x <= 5:", results)
```

### 3. Write a Python program to find the determinant of the matrix
\[ 
A =
\begin{bmatrix}
1 & 0 & 5 \\
2 & 1 & 6 \\
3 & 4 & 0 
\end{bmatrix} 
\]
and 
\[ 
B =
\begin{bmatrix}
2 & 5 \\
-1 & 4 
\end{bmatrix} 
\]

```python
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

### 1. Write a Python program to estimate the value of the integral 
\[ R_{1}^{3} \frac{1}{x} \, dx \] 
using Simpson’s \( \frac{1}{3} \) rule (n=8).

```python
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
    return 1 / x

a = 1
b = 3
n = 8
integral_value = simpson_13(a, b, n, integrand)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to evaluate the interpolated value \( f(2.7) \) of the given data
\( f(2)=0.69315, f(2.5)=0.91629, f(3)=1.09861 \).

```python
# Given data
x_points = [2, 2.5, 3]
y_points = [0.69315, 0.91629, 1.09861]

# Function to perform linear interpolation
def linear_interpolation(x_points, y_points, x_value):
    return np.interp(x_value, x_points, y_points)

# Estimate f(2.7)
interpolated_value = linear_interpolation(x_points, y_points, 2.7)
print("Interpolated value f(2.7):", interpolated_value)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to obtain the approximate real root of 
\[ x^3 - 4x - 9 = 0 \] 
using the Regula-falsi method.

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

### 2. Write a Python program to estimate the value of the integral 
\[ R_{0}^{1} \cos(x) \, dx \] 
using the Trapezoidal rule (n=5).

```python
def trapezoidal(a, b, n, f):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        result += f(a + i * h)
    
    result *= h
    return result

# Function to integrate
def f(x):
    return np.cos(x)

a = 0
b = 1
n = 5
integral_value = trapezoidal(a, b, n, f)

print("Estimated value of the integral:", integral_value)
```


