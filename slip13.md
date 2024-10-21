
# Slip No. 13

## Q.1. Attempt any two of the following. [10]

### 1. Using Python code, evaluate the following expressions for two complex numbers \( z_1 = 3 + 2j \) and \( z_2 = -4 + 1j \).
#### (a) \( z_1 + z_2 \)

```python
# Define complex numbers
z1 = 3 + 2j
z2 = -4 + 1j

# Evaluate the expressions
result_a = z1 + z2
print("z1 + z2 =", result_a)
```

#### (b) \( z_1 - z_2 \)

```python
# Evaluate the expression
result_b = z1 - z2
print("z1 - z2 =", result_b)
```

#### (c) \( z_1 \times z_2 \)

```python
# Evaluate the expression
result_c = z1 * z2
print("z1 * z2 =", result_c)
```

### 2. Use Python code to find the area and circumference of a square whose length is 5.

```python
# Length of the square
length = 5

# Calculate area and circumference
area = length ** 2
circumference = 4 * length

print("Area of square:", area)
print("Circumference of square:", circumference)
```

### 3. Write a Python program to generate the square numbers from 1 to 10.

```python
# Generate square numbers from 1 to 10
squares = [i ** 2 for i in range(1, 11)]
print("Square numbers from 1 to 10:", squares)
```

## Q.2. Attempt any two of the following. [10]

### 1. Write Python code to reverse the string \( S = [1, 2, 3, 4, 5, 6, 7, 8, 9] \).

```python
# Reverse the list S
S = [1, 2, 3, 4, 5, 6, 7, 8, 9]
reversed_S = S[::-1]
print("Reversed list S:", reversed_S)
```

### 2. Write a Python program to find \( f(x) = x^2 + 3x \) where \( -1 \leq x \leq 3 \).

```python
# Define the function
def f(x):
    return x ** 2 + 3 * x

# Calculate f(x) for x in the range -1 to 3
results = {x: f(x) for x in range(-1, 4)}
print("Values of f(x) from -1 to 3:", results)
```

### 3. Write Python code to find the average of numbers from 50 to 100.

```python
# Calculate the average of numbers from 50 to 100
numbers = list(range(50, 101))
average = sum(numbers) / len(numbers)

print("Average of numbers from 50 to 100:", average)
```

## Q.3. a. Attempt any one of the following. [7]

### 1. Write a Python program to estimate the value of the integral 
\[ R_{0}^{5} \sqrt{1 + x^3} \, dx \] 
using Simpson’s \( \frac{1}{3} \) rule (n=10).

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
    return (1 + x ** 3) ** 0.5

a = 0
b = 5
n = 10  # Must be even
integral_value = simpson_13(a, b, n, f)

print("Estimated value of the integral:", integral_value)
```

### 2. Write a Python program to evaluate the interpolated value \( f(5.5) \) of the given data.

```python
# Given data points
x_values = [3, 5, 7, 9]
y_values = [5, 7, 27, 64]

# Linear interpolation function
def linear_interpolation(x, x_values, y_values):
    for i in range(len(x_values) - 1):
        if x_values[i] <= x <= x_values[i + 1]:
            # Linear interpolation formula
            return y_values[i] + (y - x_values[i]) * (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
    return None  # If x is out of bounds

y = 5.5
interpolated_value = linear_interpolation(y, x_values, y_values)
print(f"Interpolated value f(5.5): {interpolated_value}")
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
    return x ** 3 - 4 * x - 9

a = 2
b = 3
n = 10
root = regula_falsi(a, b, f, n)
print("Approximate real root:", root)
```

### 2. Write a Python program to evaluate \( f(2.7) \) by backward difference formula of the given data.

```python
# Given data points
x_values = [1, 2, 3, 4, 5]
y_values = [40, 60, 65, 50, 18]

# Backward difference formula function
def backward_difference(x, x_values, y_values):
    n = len(y_values)
    # Calculate backward differences
    b_diff = [y_values[i] for i in range(n)]
    
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            b_diff[i] = b_diff[i] - b_diff[i - 1]
    
    # Calculate the value of f(2.7)
    h = x_values[1] - x_values[0]
    k = (x - x_values[-1]) / h
    
    value = y_values[-1]
    
    for i in range(1, n):
        term = 1
        for j in range(i):
            term *= (k + j) / (j + 1)
        value += term * b_diff[-i - 1]
    
    return value

# Evaluate f(2.7)
x_target = 2.7
value_at_target = backward_difference(x_target, x_values, y_values)
print(f"Value of f(2.7): {value_at_target}")
```

