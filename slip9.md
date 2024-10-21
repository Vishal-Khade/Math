Here’s the solution for Slip No. 9 formatted in markdown for easy readability and usability in a Python environment:

```markdown
# Slip No. 9

## Q.1. Attempt any two of the following. [10]

### 1. Using Python, evaluate each of the following expressions.
#### a. \( 30 \mod 2 + 7 - (3 + 9) \times 20 \div 5 \)

```python
# Evaluating the expression
result_a = 30 % 2 + 7 - (3 + 9) * 20 / 5
print("Result of expression a:", result_a)
```

#### b. \( 30 \times 10 \text{ floor division } 3 + 30 \mod 3 \)

```python
# Evaluating the expression
result_b = 30 * 10 // 3 + 30 % 3
print("Result of expression b:", result_b)
```

#### c. \( 5^5 - 5^3 + 7 \text{ floor division } 7 \)

```python
# Evaluating the expression
result_c = 5**5 - 5**3 + 7 // 7
print("Result of expression c:", result_c)
```

### 2. Use print command in Python to find
#### (a) sin(30)

```python
import math

# Printing sine of 30 degrees (converted to radians)
print("sin(30):", math.sin(math.radians(30)))
```

#### (b) pi

```python
# Printing the value of pi
print("pi:", math.pi)
```

#### (c) e

```python
# Printing the value of e
print("e:", math.e)
```

#### (d) cos(30)

```python
# Printing cosine of 30 degrees (converted to radians)
print("cos(30):", math.cos(math.radians(30)))
```

### 3. Write Python code to generate modulus value of -10, 10, -1, 1, 0.

```python
# List of numbers
numbers = [-10, 10, -1, 1, 0]

# Generating modulus values
modulus_values = [abs(num) for num in numbers]
print("Modulus values:", modulus_values)
```

## Q.2. Attempt any two of the following. [10]

### 1. Use Python code to generate second, fifth, and eighth characters from the string ‘MATHEMATICS’.

```python
string = "MATHEMATICS"

# Generating specified characters (index starts from 0)
characters = [string[1], string[4], string[7]]
print("Second, fifth, and eighth characters:", characters)
```

### 2. Using Python, find the eigenvalues and corresponding eigenvectors of the matrix
\[ 
\begin{bmatrix}
3 & -2 \\
6 & -4
\end{bmatrix} 
\]

```python
import numpy as np

# Define the matrix
matrix = np.array([[3, -2],
                   [6, -4]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### 3. Write Python code to verify \( (AB)^{-1} = B^{-1}A^{-1} \).

```python
# Define matrices A and B
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Calculate inverse of A and B
A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

# Calculate (AB) and its inverse
AB = np.dot(A, B)
AB_inv = np.linalg.inv(AB)

# Verify if (AB)^{-1} equals B^{-1}A^{-1}
verification = np.allclose(AB_inv, np.dot(B_inv, A_inv))
print("Verification of (AB)^{-1} = B^{-1}A^{-1}:", verification)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral 
\[ R_{1}^{10} (x^2 + 5x) \, dx \] 
using Simpson’s \( \frac{1}{3} \) rule (n=5).

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
    return x**2 + 5*x

a = 1
b = 10
n = 5
integral_value = simpson_13(a, b, n, integrand)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to evaluate the interpolated value \( f(2.5) \) of the given data.

```python
import numpy as np

# Given data
x = np.array([1, 2, 3, 4])
y = np.array([1, 8, 27, 64])

# Function to perform linear interpolation
def linear_interpolation(x_points, y_points, x_value):
    return np.interp(x_value, x_points, y_points)

# Estimate f(2.5)
interpolated_value = linear_interpolation(x, y, 2.5)
print("Interpolated value f(2.5):", interpolated_value)
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

### 2. Write a Python program to evaluate the fourth order forward difference of the given data.

```python
import numpy as np

# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([40, 60, 65, 50, 18])

# Function to calculate forward differences
def forward_difference(y):
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table

# Calculate forward differences
diff_table = forward_difference(y)
print("Forward Difference Table:\n", diff_table)
```

This structured format makes the code clear and easy to run in any Python environment.
