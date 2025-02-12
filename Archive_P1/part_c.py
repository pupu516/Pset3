import numpy as np
from scipy.linalg import inv
import sympy as sp

# Define matrix A and vector w
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

w = np.array([1, 2, 3])

# Compute inverse of A (covariance matrix Sigma)
Sigma = inv(A)

# Compute mean vector mu
mu = Sigma @ w

# Print mean vector and covariance matrix
print("Mean vector mu:")
print(mu)
print("\nCovariance matrix Sigma:")
print(Sigma)

# Function to compute moments numerically
def compute_moment(mu, Sigma, indices):
    if len(indices) == 1:
        return mu[indices[0]]
    elif len(indices) == 2:
        return mu[indices[0]] * mu[indices[1]] + Sigma[indices[0], indices[1]]
    elif len(indices) == 3:
        i, j, k = indices
        return mu[i] * mu[j] * mu[k] + mu[i] * Sigma[j, k] + mu[j] * Sigma[i, k] + mu[k] * Sigma[i, j]
    elif len(indices) == 4:
        i, j, k, l = indices
        return (mu[i] * mu[j] * mu[k] * mu[l] +
                mu[i] * mu[j] * Sigma[k, l] +
                mu[i] * mu[k] * Sigma[j, l] +
                mu[i] * mu[l] * Sigma[j, k] +
                mu[j] * mu[k] * Sigma[i, l] +
                mu[j] * mu[l] * Sigma[i, k] +
                mu[k] * mu[l] * Sigma[i, j] +
                Sigma[i, j] * Sigma[k, l] +
                Sigma[i, k] * Sigma[j, l] +
                Sigma[i, l] * Sigma[j, k])
    else:
        raise ValueError("Moment calculation not implemented for this order.")

# Compute moments numerically
numerical_moments = {
    "<v1>": compute_moment(mu, Sigma, [0]),
    "<v2>": compute_moment(mu, Sigma, [1]),
    "<v3>": compute_moment(mu, Sigma, [2]),
    "<v1 v2>": compute_moment(mu, Sigma, [0, 1]),
    "<v2 v3>": compute_moment(mu, Sigma, [1, 2]),
    "<v1 v3>": compute_moment(mu, Sigma, [0, 2]),
    "<v1^2 v2>": compute_moment(mu, Sigma, [0, 0, 1]),
    "<v2 v3^2>": compute_moment(mu, Sigma, [1, 2, 2]),
    "<v1^2 v2^2>": compute_moment(mu, Sigma, [0, 0, 1, 1]),
    "<v2^2 v3^2>": compute_moment(mu, Sigma, [1, 1, 2, 2]),
}

# Print numerical moments
print("\nNumerical Moments:")
for key, value in numerical_moments.items():
    print(f"{key}: {value}")

# Define symbolic variables for closed-form expressions
A_sym = sp.MatrixSymbol('A', 3, 3)
w_sym = sp.MatrixSymbol('w', 3, 1)

# Compute A inverse symbolically
A_inv_sym = A_sym**-1

# Compute mean vector mu symbolically
mu_sym = A_inv_sym @ w_sym

# Define symbolic Sigma (covariance matrix)
Sigma_sym = A_inv_sym

# Function to compute moments symbolically
def symbolic_moment(A, w, indices):
    if len(indices) == 1:
        return mu_sym[indices[0]]
    elif len(indices) == 2:
        i, j = indices
        return mu_sym[i] * mu_sym[j] + Sigma_sym[i, j]
    elif len(indices) == 3:
        i, j, k = indices
        return mu_sym[i] * mu_sym[j] * mu_sym[k] + mu_sym[i] * Sigma_sym[j, k] + mu_sym[j] * Sigma_sym[i, k] + mu_sym[k] * Sigma_sym[i, j]
    elif len(indices) == 4:
        i, j, k, l = indices
        return (mu_sym[i] * mu_sym[j] * mu_sym[k] * mu_sym[l] +
                mu_sym[i] * mu_sym[j] * Sigma_sym[k, l] +
                mu_sym[i] * mu_sym[k] * Sigma_sym[j, l] +
                mu_sym[i] * mu_sym[l] * Sigma_sym[j, k] +
                mu_sym[j] * mu_sym[k] * Sigma_sym[i, l] +
                mu_sym[j] * mu_sym[l] * Sigma_sym[i, k] +
                mu_sym[k] * mu_sym[l] * Sigma_sym[i, j] +
                Sigma_sym[i, j] * Sigma_sym[k, l] +
                Sigma_sym[i, k] * Sigma_sym[j, l] +
                Sigma_sym[i, l] * Sigma_sym[j, k])
    else:
        raise ValueError("Moment calculation not implemented for this order.")

# Compute symbolic moments
symbolic_moments = {
    "<v1>": symbolic_moment(A_sym, w_sym, [0]),
    "<v2>": symbolic_moment(A_sym, w_sym, [1]),
    "<v3>": symbolic_moment(A_sym, w_sym, [2]),
    "<v1 v2>": symbolic_moment(A_sym, w_sym, [0, 1]),
    "<v2 v3>": symbolic_moment(A_sym, w_sym, [1, 2]),
    "<v1 v3>": symbolic_moment(A_sym, w_sym, [0, 2]),
    "<v1^2 v2>": symbolic_moment(A_sym, w_sym, [0, 0, 1]),
    "<v2 v3^2>": symbolic_moment(A_sym, w_sym, [1, 2, 2]),
    "<v1^2 v2^2>": symbolic_moment(A_sym, w_sym, [0, 0, 1, 1]),
    "<v2^2 v3^2>": symbolic_moment(A_sym, w_sym, [1, 1, 2, 2]),
}

# Print symbolic moments
print("\nClosed-Form Moments in Terms of A and w:")
for key, value in symbolic_moments.items():
    print(f"{key}: {value}")

# Verify numerically by substituting A and w into symbolic expressions
A_val = sp.Matrix(A)
w_val = sp.Matrix(w)

# Substitute A and w into symbolic expressions
verified_moments = {
    key: value.subs({A_sym: A_val, w_sym: w_val}).evalf()
    for key, value in symbolic_moments.items()
}

# Print verified moments
print("\nVerified Moments (Numerical Substitution):")
for key, value in verified_moments.items():
    print(f"{key}: {value}")
