using LinearAlgebra

"""
    compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

Build the 2n x 2n Hamiltonian matrix H for Vaughan's method,
from the standard-form LQ matrices (no discount, no cross-term).

The eigenvalues of H come in reciprocal pairs (mu, 1/mu).

# Arguments
- `A_tilde` : n x n  standard-form state transition
- `B_tilde` : n x m  standard-form control effect
- `Q_tilde` : n x n  standard-form state cost
- `R`       : m x m  control cost

# Returns
- `H` : 2n x 2n  Hamiltonian matrix
"""
function compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)
    Ainv = inv(A_tilde)
    M = Ainv * B_tilde * (R \ B_tilde')       # A^{-1} B R^{-1} B'
    H = [Ainv            M;
         Q_tilde*Ainv    Q_tilde*M + A_tilde']
    return H
end


"""
    solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)

Solve the LQ problem via Vaughan's (1970) eigenvalue decomposition
of the Hamiltonian matrix H.

# Algorithm
1. Build H from the standard-form matrices.
2. Eigendecompose H.  Its 2n eigenvalues come in reciprocal pairs
   (mu_i, 1/mu_i).  Select the n eigenvectors whose eigenvalues
   have |mu| > 1  (the "unstable" eigenvalues of H, which correspond
   to the *stable* eigenvalues of the forward system H^{-1}).
3. Stack those eigenvectors as columns:
       V = [V_11; V_21]   (each n x n)
   Then the value-function matrix is  P = V_21 * V_11^{-1}.
4. Recover the feedback:
       F_tilde = R^{-1} B_tilde' P
       F       = F_tilde + R^{-1} W'     (undo cross-term)

# Arguments
- `Q_tilde` : n x n
- `R`       : m x m
- `A_tilde` : n x n
- `B_tilde` : n x m
- `W`       : n x m  original cross-term (needed to recover F)

# Returns
- `F` : m x n  feedback matrix in original coordinates
- `P` : n x n  value-function matrix
"""
function solve_vaughan(Q_tilde, R, A_tilde, B_tilde, W)
    n = size(A_tilde, 1)

    H = compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

    eig = eigen(H)
    vals = eig.values
    vecs = eig.vectors

    # Select eigenvectors with |eigenvalue| > 1 (unstable for H,
    # stable for forward system H^{-1})
    idx = sortperm(abs.(vals), rev=true)
    idx_unstable = idx[1:n]

    V = vecs[:, idx_unstable]
    V11 = V[1:n,     :]
    V21 = V[n+1:2n,  :]

    P = real.(V21 / V11)
    P = (P + P') / 2                            # enforce symmetry

    F_tilde = (R + B_tilde' * P * B_tilde) \ (B_tilde' * P * A_tilde)
    F = real.(F_tilde + R \ W')

    println("Vaughan: eigenvalues of H = ", round.(sort(abs.(vals), rev=true), digits=4))
    return F, P
end
