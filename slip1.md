
# Slip No. 1
### Q1.1
**Evaluate the following expressions:**

- a. \( 20 \mod 2 + 7 - \left(3 + 7\right) \times 20 \div 2 \)
- b. \( 30 \times 10 \text{ floor division } 3 + 10 \mod 3 \)
- c. \( 25 - 24 + 4 \text{ floor division } 4 \)

```python
# Expression a
result_a = 20 % 2 + 7 - (3 + 7) * 20 / 2
print("Result a:", result_a)

# Expression b
result_b = 30 * 10 // 3 + 10 % 3
print("Result b:", result_b)

# Expression c
result_c = 25 - 24 + 4 // 4
print("Result c:", result_c)
```

### Q1.2
**Repeat the following strings 9 times using the string operator `*`.**

- a. Python
- b. Mathematics

```python
# Repeating strings
string_a = "Python" * 9
string_b = "Mathematics" * 9
print(string_a)
print(string_b)
```

### Q1.3
**Write a Python program to generate the square of numbers from 1 to 10.**

```python
# Generating squares of numbers from 1 to 10
squares = [i**2 for i in range(1, 11)]
print("Squares from 1 to 10:", squares)
```

---

### Q2.1
**Construct the following matrices:**

1. Identity matrix of order 10 × 10.
2. Zero matrix of order 7 × 3.
3. Ones matrix of order 5 × 4.

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

### Q2.2
**Write a Python program to find the 10th term of the sequence of function \( f(x) = x^2 + x \).**

```python
def f(x):
    return x**2 + x

# 10th term
term_10 = f(10)
print("10th term of the sequence f(x) = x^2 + x is:", term_10)
```

### Q2.3
**Generate all the prime numbers between 1 to 100 using Python code.**

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

---

### Q3.1
**Estimate the value of the integral \( \int_0^{\pi} \sin(x) dx \) using Simpson’s 1/3rd rule (\( n=6 \)).**

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
print("Estimated value of integral using Simpson’s 1/3rd rule:", result)
```

### Q3.2
**Evaluate interpolated value \( f(3) \) of the given data using Lagrange's method.**

- Data:
  - \( x = [0, 1, 2, 5] \)
  - \( y = [5, 13, 22, 129] \)

```python
def lagrange_interpolation(x_values, y_values, x_to_find):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term = term * (x_to_find - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

x_values = [0, 1, 2, 5]
y_values = [5, 13, 22, 129]

# Interpolating f(3)
f_3 = lagrange_interpolation(x_values, y_values, 3)
print("Interpolated value f(3):", f_3)
```

---

This completes **Slip No. 1**.

Let me know when you're ready to proceed with **Slip No. 2**!
