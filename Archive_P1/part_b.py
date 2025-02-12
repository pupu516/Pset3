import numpy as np
from scipy.integrate import nquad

def compute_numeric_integral(A, w):

    # Convert inputs to numpy arrays and validate

    A = np.array(A, dtype=np.float64)
    w = np.array(w, dtype=np.float64).flatten()
    N = len(w)
    
    # Validate matrix dimensions
    if A.shape != (N, N):
        raise ValueError(f"Matrix A must be {N}x{N} for vector w of length {N}")
        
    # Define the integrand function
    def integrand(*args):
        v = np.array(args)
        quadratic_term = -0.5 * np.sum(v @ A @ v)  # -½ vᵀAv
        linear_term = np.dot(v, w)                # vᵀw
        return np.exp(quadratic_term + linear_term)
    
    # Set up infinite integration limits for all dimensions
    limits = [(-np.inf, np.inf) for _ in range(N)]
    
    # Compute the numerical integral
    result, _ = nquad(integrand, limits)
    return result

def compute_closed_form(A, w):

    # Convert inputs to numpy arrays and validate
    A = np.array(A, dtype=np.float64)
    w = np.array(w, dtype=np.float64).flatten()
    N = len(w)
    
    # Validate matrix dimensions
    if A.shape != (N, N):
        raise ValueError(f"Matrix A must be {N}x{N} for vector w of length {N}")
    
    try:
        # Compute inverse of A
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix A is singular and cannot be inverted")
    
    # Compute determinant of A
    det_A = np.linalg.det(A)
    
    # Compute normalization term: √((2π)^N / det(A))
    normalization = np.sqrt(((2 * np.pi) ** N) / det_A)
    
    # Compute quadratic form: ½ wᵀ A⁻¹ w
    quadratic_form = 0.5 * w.T @ A_inv @ w
    
    # Compute the closed-form expression
    closed_form = normalization * np.exp(quadratic_form)
    return closed_form


# Define matrices A and A', and vector w
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]], dtype=np.float64)

A_prime = np.array([[4, 2, 1],
                    [2, 1, 3],
                    [1, 3, 6]], dtype=np.float64)

w = np.array([1, 2, 3], dtype=np.float64)

# Compute numerical integrals
numerical_A = compute_numeric_integral(A, w)
numerical_A_prime = compute_numeric_integral(A_prime, w)

# Compute closed-form expressions
closed_form_A = compute_closed_form(A, w)
closed_form_A_prime = compute_closed_form(A_prime, w)

# Print results
print("Results for Matrix A:")
print(f"Numerical Integral: {numerical_A:.6f}")
print(f"Closed-form: {closed_form_A:.6f}")
print(f"Discrepancy: {abs(numerical_A - closed_form_A):.6f}")

print("\nResults for Matrix A':")
print(f"Numerical Integral: {numerical_A_prime:.6f}")
print(f"Closed-form: {closed_form_A_prime:.6f}")
print(f"Discrepancy: {abs(numerical_A_prime - closed_form_A_prime):.6f}")


