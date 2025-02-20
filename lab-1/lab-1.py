import time
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal, InvalidOperation

def matrix_fib(n):
    def matrix_mult(A, B):
        return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]
    
    def matrix_pow(M, power):
        result = [[1, 0], [0, 1]]
        base = M
        while power:
            if power % 2 == 1:
                result = matrix_mult(result, base)
            base = matrix_mult(base, base)
            power //= 2
        return result
    
    F = [[1, 1], [1, 0]]
    if n <= 1:
        return n
    return matrix_pow(F, n - 1)[0][0]

def dp_fib(n):
    l1 = [0, 1]
    for i in range(2, n + 1):
        l1.append(l1[i - 1] + l1[i - 2])
    return l1[n]

def recursive_fib(n):
    if n <= 1:
        return n
    return recursive_fib(n - 1) + recursive_fib(n - 2)

def fibonacci_binet(n):
    try:
        sqrt_5 = Decimal(5).sqrt()
        phi = (Decimal(1) + sqrt_5) / Decimal(2)
        return int((phi**n / sqrt_5).quantize(Decimal(1)))
    except InvalidOperation:
        return None

def fast_doubling_fib(n):
    def fib(n):
        if n == 0:
            return (0, 1)
        else:
            a, b = fib(n // 2)
            c = a * (2 * b - a)
            d = b * b + a * a
            if n % 2 == 0:
                return (c, d)
            else:
                return (d, c + d)

    return fib(n)[0]

def iterative_fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def measure_time():
    limited_scope = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35]
    large_scope = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
    
    def time_method(method, n, iterations=3):
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            method(n)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return times

    rec_times = {n: time_method(recursive_fib, n) for n in limited_scope}
    binet_times = {n: time_method(fibonacci_binet, n) for n in large_scope}
    dp_times = {n: time_method(dp_fib, n) for n in large_scope}
    mat_times = {n: time_method(matrix_fib, n) for n in large_scope}
    fast_doubling_times = {n: time_method(fast_doubling_fib, n) for n in large_scope}
    iterative_times = {n: time_method(iterative_fib, n) for n in large_scope}

    print("Unoptimized Recursive Method:\n", pd.DataFrame(rec_times, index=[f"Run {i+1}" for i in range(3)]))
    print("\nBinet Method:\n", pd.DataFrame(binet_times, index=[f"Run {i+1}" for i in range(3)]))
    print("\nDynamic Programming Method:\n", pd.DataFrame(dp_times, index=[f"Run {i+1}" for i in range(3)]))
    print("\nMatrix Power Method:\n", pd.DataFrame(mat_times, index=[f"Run {i+1}" for i in range(3)]))
    print("\nFast Doubling Method:\n", pd.DataFrame(fast_doubling_times, index=[f"Run {i+1}" for i in range(3)]))
    print("\nIterative Method:\n", pd.DataFrame(iterative_times, index=[f"Run {i+1}" for i in range(3)]))

    max_y = max(max(binet_times[n][0] for n in large_scope), max(dp_times[n][0] for n in large_scope),
                max(mat_times[n][0] for n in large_scope), max(fast_doubling_times[n][0] for n in large_scope),
                max(iterative_times[n][0] for n in large_scope)) / 2

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    axs[0, 0].plot(limited_scope, [rec_times[n][0] for n in limited_scope], label='Unoptimized Recursive', color='blue', marker='o')
    axs[0, 0].set_xlabel('n-th Fibonacci')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].set_title('Unoptimized Recursive Time')
    axs[0, 0].grid(True)

    for i, (method_times, title, color) in enumerate(zip(
            [binet_times, dp_times, mat_times, fast_doubling_times, iterative_times],
            ['Binet Method', 'Dynamic Programming', 'Matrix Power', 'Fast Doubling', 'Iterative Method'],
            ['purple', 'green', 'red', 'orange', 'brown'])):
        ax = axs[(i + 1) // 2, (i + 1) % 2]
        ax.plot(large_scope, [method_times[n][0] for n in large_scope], label=title, color=color, marker='x')
        ax.set_xlabel('n-th Fibonacci')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{title} Time')
        ax.grid(True)
        ax.set_ylim([0, max_y])

    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(limited_scope, [rec_times[n][0] for n in limited_scope], label='Unoptimized Recursive', color='blue', marker='o')
    plt.plot(large_scope, [binet_times[n][0] for n in large_scope], label='Binet Method', color='purple', marker='x')
    plt.plot(large_scope, [dp_times[n][0] for n in large_scope], label='Dynamic Programming', color='green', marker='x')
    plt.plot(large_scope, [mat_times[n][0] for n in large_scope], label='Matrix Power', color='red', marker='s')
    plt.plot(large_scope, [fast_doubling_times[n][0] for n in large_scope], label='Fast Doubling', color='orange', marker='^')
    plt.plot(large_scope, [iterative_times[n][0] for n in large_scope], label='Iterative Method', color='brown', marker='v')
    plt.xlabel('n-th Fibonacci')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of All Methods')
    plt.grid(True)
    plt.ylim([0, max_y])
    plt.legend()
    plt.show()

measure_time()
