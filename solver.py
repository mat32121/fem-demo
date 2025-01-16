# Mateusz Kosman
# Problem: 4.5 Potencjał elektromagnetyczny

from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

# Computes integral of function fun from a to b
def integral(fun: Callable[[float], float], a: float, b: float) -> float:
    if a == b: return 0.0

    sample = np.linspace(a, b, NUM_INTEGRAL_SAMPLES)
    area = 0.0
    for (x1, x2) in zip(sample[:-1], sample[1:]):
        area += (fun(x2)+fun(x1))*(x2-x1)/2

    return area

# Basis function e described in the notes
def e(i: int, x: float) -> float:
    assert 0 <= i and i <= N
    if i == 0:
        if X[0] <= x and x < X[1]:
            return 1-x/X[1]
    elif i == N:
        if X[N-1] <= x and x <= X[N]:
            return (x-X[N-1])/(X[N]-X[N-1])
    elif X[i-1] < x and x <= X[i]:
        return (x-X[i-1])/(X[i]-X[i-1])
    elif X[i] < x and x < X[i+1]:
        return (x-X[i+1])/(X[i]-X[i+1])
    return 0.0

# Derivative of basis function e
def eprime(i: int, x: float) -> float:
    assert 0 <= i and i <= N
    if i == 0:
        if X[0] < x and x < X[1]:
            return -1.0
    elif i == N:
        if X[N-1] < x and x < X[N]:
            return 1.0
    elif X[i-1] < x and x < X[i]:
        return 1.0
    elif X[i] < x and x < X[i+1]:
        return -1.0
    return 0.0

def binsearch(x: float) -> int:
    l, r = 0, N
    while l < r:
        m = (l+r)//2
        if X[m+1] < x:
            l = m+1
        else:
            r = m
    return l

# Only two "neighboring" base functions have non-zero values at x
def phi(x: float) -> float:
    if x < X[0] or X[N] < x:
        return None # Undefined outside of the domain
    i = binsearch(x)
    return W[i]*e(i, x) + W[i+1]*e(i+1, x)

if __name__ == '__main__':
    NUM_PHI_SAMPLES = 10000
    NUM_INTEGRAL_SAMPLES = 1000

    N = int(argv[1])
    assert N > 0
    DOMAIN = [0.0, 3.0]
    RHO = 1
    EPSILON_R = [(10, (0.0, 1.0)), (5, (1.0, 2.0)), (1, (2.0, 3.0))]

    X = np.linspace(*DOMAIN, N+1)
    B = np.zeros((N, N))
    B2 = np.zeros(N)
    L = np.zeros(N)

    # For the given basis the integral is equal to 0 if |i-j| >= 2
    INTEGRAL = [[integral(lambda x: eprime(i, x)*eprime(j, x), *DOMAIN) if abs(i-j) < 2 else 0.0 for j in range(i+1)] for i in range(N+1)]

    # Computing B(e_i, e_j)
    B[0][0] = e(0, 0.0)**2 - INTEGRAL[0][0]
    for i in range(N):
        B[i][i] = e(i, 0.0)**2 - INTEGRAL[i][i]
    for i in range(1, N):
        for j in range(i):
            B[i][j] = B[j][i] = e(i, 0.0)*e(j, 0.0) - INTEGRAL[i][j]

    # Computing B(2e_n, e_j)
    for j in range(N):
        B2[j] = 2*e(N, 0.0)*e(j, 0.0) - 2*INTEGRAL[N][j]
    
    # Computing L(e_j)
    current_segment = 0
    for j in range(N):
        if EPSILON_R[current_segment][1][1] < X[j]:
            current_segment += 1
        L[j] = 5*e(j, 0.0) - RHO/EPSILON_R[current_segment][0]*integral(lambda x: e(j, x), *DOMAIN)

    # Computing the inverse matrix of B
    B_inv = np.linalg.inv(B)
    # Computing W
    W = B_inv @ (L-B2)
    # Assuming w_n = 2
    W = np.append(W, 2)

    # Sampling phi
    X_SAMPLES = np.linspace(*DOMAIN, NUM_PHI_SAMPLES)
    Y_SAMPLES = np.array([phi(x) for x in X_SAMPLES])

    fig, ax = plt.subplots()
    ax.set_title("Integrals computed numerically")
    ax.plot(X_SAMPLES, Y_SAMPLES)
    # plt.show()

    # ------------------------------------------------
    # Precomputed integrals
    B = np.zeros((N, N))
    B2 = np.zeros(N)
    L = np.zeros(N)

    B[0][0] = 1-3/N
    for i in range(1, N):
        B[i][i] = -6/N
    for i in range(N-1):
        B[i][i+1] = B[i+1][i] = 3/N

    B2[N-1] = 6/N

    L[0] = 5-3*RHO/(2*N*EPSILON_R[0][0])
    current_segment = 0
    for j in range(1, N):
        if EPSILON_R[current_segment][1][1] < X[j]:
            current_segment += 1
        L[j] = -3*RHO/(N*EPSILON_R[current_segment][0])

    # Computing the inverse matrix of B
    B_inv = np.linalg.inv(B)
    # Computing W
    W = B_inv @ (L-B2)
    # Assuming w_n = 2
    W = np.append(W, 2)

    # Sampling phi
    Y_SAMPLES = np.array([phi(x) for x in X_SAMPLES])

    fig, ax = plt.subplots()
    ax.set_title("Precomputed integrals")
    ax.plot(X_SAMPLES, Y_SAMPLES)
    plt.show()