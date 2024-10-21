
# Slip No. 7

## Q.1. Attempt any two of the following. [10]

### 1. Using Python, evaluate the following expressions of two complex numbers \( z_1 = 5 + 3j \) and \( z_2 = -5 + 7j \).

```python
z1 = 5 + 3j
z2 = -5 + 7j

# a. z1 + z2
sum_result = z1 + z2
print("z1 + z2 =", sum_result)

# b. z1 - z2
difference_result = z1 - z2
print("z1 - z2 =", difference_result)

# c. z1 * z2
product_result = z1 * z2
print("z1 * z2 =", product_result)
```

### 2. Repeat the following strings 7 times using the string operator ‘*’ in Python.

```python
# a. Complex Number
complex_string = "Complex Number " * 7
print(complex_string)

# b. Real Number
real_string = "Real Number " * 7
print(real_string)
```

### 3. Write Python code to generate the cube of numbers from 1 to 50.

```python
cubes = [x**3 for x in range(1, 51)]
print("Cubes of numbers from 1 to 50:", cubes)
```

## Q.2. Attempt any two of the following. [10]

### 1. Using Python code, construct a ones matrix of order \( 10 \times 10 \) and hence find the determinant, trace, and transpose of it.

```python
import numpy as np

# Constructing ones matrix of order 10
ones_matrix = np.ones((10, 10))

# Finding determinant, trace, and transpose
determinant = np.linalg.det(ones_matrix)
trace = np.trace(ones_matrix)
transpose = ones_matrix.T

print("Ones Matrix (10x10):\n", ones_matrix)
print("Determinant:", determinant)
print("Trace:", trace)
print("Transpose of Ones Matrix:\n", transpose)
```

### 2. Write Python code to obtain \( f(-1) \), \( f(0) \), \( f(1) \) of the function \( f(x) = x^3 - 4x - 9 \).

```python
def f(x):
    return x**3 - 4*x - 9

# Evaluating the function at -1, 0, and 1
f_neg1 = f(-1)
f_0 = f(0)
f_1 = f(1)

print("f(-1):", f_neg1)
print("f(0):", f_0)
print("f(1):", f_1)
```

### 3. Generate all the prime numbers between 500 to 1000 using a Python program.

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

prime_numbers = [num for num in range(500, 1001) if is_prime(num)]
print("Prime numbers between 500 and 1000:", prime_numbers)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral \( \int_1^5 x^3 \, dx \) using Simpson’s \( \frac{1}{3} \) rule (n=6).

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
    return x**3

a = 1
b = 5
n = 6
integral_value = simpson_13(a, b, n, integrand)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to evaluate the interpolated value \( f(3) \) of the given data.

```python
import numpy as np

# Given data points
x = np.array([0, 1, 2, 5])
y = np.array([2, 3, 12, 147])

# Interpolating value for x = 3
interpolated_value = np.interp(3, x, y)
print("Interpolated value f(3):", interpolated_value)
```

## b. Attempt any one of the following. [8]

### 1. Write a Python program to estimate the value of the integral \( \int_2^{10} \frac{1}{1+x} \, dx \) using the Trapezoidal rule (n=8).

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

a = 2
b = 10
n = 8
integral_value = trapezoidal(a, b, n, integrand)
print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to evaluate \( f(2.8) \) using the backward difference formula of the given data.

```python
# Given data points
x = np.array([0, 1, 2, 3])
y = np.array([1, 0, 1, 10])

# Function to calculate backward difference
def backward_difference(x, y, value):
    n = len(y)
    h = x[1] - x[0]
    
    # Backward difference formula
    f_value = y[n-1]
    
    for i in range(1, n):
        coeff = (-1)**i * np.prod([1/(h * (x[n-1] - x[n-1-j])) for j in range(1, i+1)]) # Product for coefficients
        f_value += coeff * y[n-1-i]
    
    return f_value

result = backward_difference(x, y, 2.8)
print("Value of f(2.8):", result)
```

