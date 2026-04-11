module ECON8208Tools

using LinearAlgebra
using Random

export mean,
       std,
       cor,
       hp_filter,
       load_raw_data,
       compute_data_moments,
       compute_model_moments,
       numerical_derivative,
       numerical_jacobian,
       numerical_hessian,
       numerical_cross_hessian,
       fixed_point_a,
       fixed_point_b,
       fixed_point_c,
       fixed_point_d,
       solve_l_star,
       solve_hs_star,
       lq_approximation,
       lq_approximation_with_aggregates,
       construct_distorted_lq_system,
       solve_riccati,
       transform_lq_matrices,
       solve_modified_riccati_equilibrium,
       compute_distorted_vaughan_H,
       solve_vaughan_distorted,
       quadratic_approximation,
       normcdf,
       tauchen,
       steady_state_system,
       solve_steady_state_newton,
       solve_hw6_steady_state,
       is_feasible_static_problem,
       labor_foc,
       solve_h_star,
       compute_h_star,
       flow_utility,
       solve_vfi,
       solve_pfi_howard,
       compute_vaughan_H,
       solve_vaughan,
       solve_lq_policy_functions,
       recover_original_policy_functions_lq,
       simulate_lq_growth_model


# -------------------------------------------------------
# Sample mean
# Input:
#   x : vector
# Output:
#   arithmetic average of x
# -------------------------------------------------------
function mean(x)
    return sum(x) / length(x)
end

# -------------------------------------------------------
# Sample standard deviation
# Input:
#   x : vector
# Output:
#   standard deviation of x using denominator T
# -------------------------------------------------------
function std(x)
    xbar = mean(x)
    return sqrt(sum((x .- xbar).^2) / length(x))
end

# -------------------------------------------------------
# Sample correlation
# Input:
#   x, y : vectors of the same length
# Output:
#   sample correlation between x and y
# -------------------------------------------------------
function cor(x, y)
    xbar = mean(x)
    ybar = mean(y)
    xdev = x .- xbar
    ydev = y .- ybar
    return sum(xdev .* ydev) / sqrt(sum(xdev.^2) * sum(ydev.^2))
end

# -------------------------------------------------------
# Numerical derivative using central difference
# Input:
#   f: R -> R
#   x: scalar
#   h: step size for finite difference (default: 1e-6)
# Output:
#   f'(x): scalar
# -------------------------------------------------------
function numerical_derivative(f, x; h=1e-6)
    return (f(x + h) - f(x - h)) / (2.0 * h)
end

# -------------------------------------------------------
# Numerical Jacobian using central difference
# Input:
#   f: R^n -> R^m
#   x: n-dimensional vector
#   h: step size for finite difference (default: 1e-6)
# Output:
#   J_f(x): m x n matrix
# -------------------------------------------------------
function numerical_jacobian(f, x; h=1e-6)
    x = Float64.(collect(x))
    fx = Float64.(collect(f(x)))

    m = length(fx)
    n = length(x)
    J = zeros(m, n)

    for j in 1:n
        e = zeros(n)
        e[j] = 1.0

        f_plus = Float64.(collect(f(x + h * e)))
        f_minus = Float64.(collect(f(x - h * e)))

        J[:, j] = (f_plus - f_minus) / (2.0 * h)
    end

    return J
end

# -------------------------------------------------------
# Numerical Hessian using second-order centered differences
# Input:
#   f: R^n -> R
#   x: n-dimensional vector
#   h: step size for finite difference (default: 1e-6)
# Output:
#   H_f(x): n x n matrix
# -------------------------------------------------------
function numerical_hessian(f, x; h=1e-6)
    x = Float64.(collect(x))
    n = length(x)
    H = zeros(n, n)

    f0 = f(x)

    for i in 1:n
        ei = zeros(n)
        ei[i] = 1.0

        H[i, i] = (f(x + h * ei) - 2.0 * f0 + f(x - h * ei)) / h^2

        for j in (i + 1):n
            ej = zeros(n)
            ej[j] = 1.0

            value = (
                f(x + h * ei + h * ej)
                - f(x + h * ei - h * ej)
                - f(x - h * ei + h * ej)
                + f(x - h * ei - h * ej)
            ) / (4.0 * h^2)

            H[i, j] = value
            H[j, i] = value
        end
    end

    return H
end

# -------------------------------------------------------
# Numerical cross Hessian using second-order centered differences
# Input:
#   f: R^n x R^m -> R
#   x: n-dimensional vector
#   u: m-dimensional vector
#   h: step size for finite difference (default: 1e-6)
# Output:
#   H_cross(x, u): n x m matrix
# -------------------------------------------------------
function numerical_cross_hessian(f, x, u; h=1e-6)
    x = Float64.(collect(x))
    u = Float64.(collect(u))

    n = length(x)
    m = length(u)
    H = zeros(n, m)

    for i in 1:n
        ei = zeros(n)
        ei[i] = 1.0

        for j in 1:m
            ej = zeros(m)
            ej[j] = 1.0

            H[i, j] = (
                f(x + h * ei, u + h * ej)
                - f(x + h * ei, u - h * ej)
                - f(x - h * ei, u + h * ej)
                + f(x - h * ei, u - h * ej)
            ) / (4.0 * h^2)
        end
    end

    return H
end

# -------------------------------------------------------
# (a) x, f(x) are scalars and f'(x) is analytically known
# Newton method for g(x) = f(x) - x = 0
# -------------------------------------------------------
function fixed_point_a(f, fp, x0; tol=1e-10, max_iter=1000)
    x = Float64(x0)

    for iter in 1:max_iter
        g = f(x) - x
        gp = fp(x) - 1.0

        if abs(g) < tol
            return x, iter, true
        end

        x = x - g / gp
    end

    return x, max_iter, false
end

# -------------------------------------------------------
# (b) x, f(x) are scalars and f'(x) is not known
# Newton method for g(x) = f(x) - x = 0
# -------------------------------------------------------
function fixed_point_b(f, x0; tol=1e-10, max_iter=1000, h=1e-6)
    x = Float64(x0)

    for iter in 1:max_iter
        g = f(x) - x
        fp_approx = numerical_derivative(f, x; h=h)
        gp = fp_approx - 1.0

        if abs(g) < tol
            return x, iter, true
        end

        x = x - g / gp
    end

    return x, max_iter, false
end

# -------------------------------------------------------
# (c) x, f(x) are vectors and the derivatives of f are known
# Newton method for g(x) = f(x) - x = 0
# -------------------------------------------------------
function fixed_point_c(f, fp, x0; tol=1e-10, max_iter=1000)
    x = Float64.(collect(x0))
    n = length(x)
    I_n = Matrix{Float64}(I, n, n)

    for iter in 1:max_iter
        g = f(x) - x

        if norm(g, Inf) < tol
            return x, iter, true
        end

        gp = fp(x) - I_n
        dx = gp \ g
        x = x - dx
    end

    return x, max_iter, false
end

# -------------------------------------------------------
# (d) x, f(x) are vectors and derivatives are not known
# Newton method for g(x) = f(x) - x = 0
# -------------------------------------------------------
function fixed_point_d(f, x0; tol=1e-10, max_iter=1000, h=1e-6)
    x = Float64.(collect(x0))
    n = length(x)
    I_n = Matrix{Float64}(I, n, n)

    for iter in 1:max_iter
        g = f(x) - x

        if norm(g, Inf) < tol
            return x, iter, true
        end

        gp = numerical_jacobian(f, x; h=h) - I_n
        dx = gp \ g
        x = x - dx
    end

    return x, max_iter, false
end

# -------------------------------------------------------
# 2(a): solve for l* in
# g(l) = u_l(c,l) / u_c(c,l) - z * f_h(k, 1-l) = 0
# Input: c, z, k, u(c,l), f(k,h), initial guess l0, tol,
# max_iter, and h for numerical differentiation.
# Output: l*, number of iterations, convergence status
# -------------------------------------------------------
function solve_l_star(c, z, k, u, f, l0; tol=1e-10, max_iter=1000, h=1e-6)
    l = Float64(l0)

    function u_c(cval, lval)
        w(x) = u(x, lval)
        return numerical_derivative(w, cval; h=h)
    end

    function u_l(cval, lval)
        w(x) = u(cval, x)
        return numerical_derivative(w, lval; h=h)
    end

    function f_h(kval, hval)
        w(x) = f(kval, x)
        return numerical_derivative(w, hval; h=h)
    end

    function g(lval)
        hval = 1.0 - lval

        uc_val = u_c(c, lval)
        ul_val = u_l(c, lval)
        fh_val = f_h(k, hval)

        if abs(uc_val) < eps(Float64)
            return NaN
        end

        return ul_val / uc_val - z * fh_val
    end

    if l0 < 0.0 || l0 > 1.0
        return l0, 0, false
    end

    if abs(g(l0)) < tol
        return l0, 0, true
    end

    for iter in 1:max_iter
        g_val = g(l)

        if !isfinite(g_val)
            return l, iter, false
        end

        gp_val = numerical_derivative(g, l; h=h)

        if !isfinite(gp_val) || abs(gp_val) < eps(Float64)
            return l, iter, false
        end

        l_new = l - g_val / gp_val

        if l_new < 0.0 || l_new > 1.0
            return l_new, iter, false
        end

        if abs(g(l_new)) < tol
            return l_new, iter + 1, true
        end

        l = l_new
    end

    return l, max_iter, false
end

# -------------------------------------------------------
# 2(b): solve for (h*, s*) in the system of equations
# G(h,s) = [g1(h,s); g2(h,s)].
# Input: w, y, u(c,l), p(s), initial guess x0 = (h0, s0),
# tol, max_iter, and h for numerical differentiation.
# Output: (h*, s*), number of iterations, convergence status
# -------------------------------------------------------
function solve_hs_star(w, y, u, p, x0; tol=1e-10, max_iter=1000, h=1e-6)
    x = Float64.(collect(x0))

    function is_feasible(xvec)
        h_work = xvec[1]
        s_shop = xvec[2]
        l = 1.0 - h_work - s_shop

        if h_work < 0.0 || s_shop < 0.0 || l < 0.0
            return false
        end

        p_val = p(s_shop)
        return isfinite(p_val) && p_val > 0.0
    end

    function u_c(cval, lval)
        w_tmp(z) = u(z, lval)
        return numerical_derivative(w_tmp, cval; h=h)
    end

    function u_l(cval, lval)
        w_tmp(z) = u(cval, z)
        return numerical_derivative(w_tmp, lval; h=h)
    end

    p_prime(sval) = numerical_derivative(p, sval; h=h)
    c_of(h_work, s_shop) = (w * h_work + y) / p(s_shop)
    l_of(h_work, s_shop) = 1.0 - h_work - s_shop

    function G(xvec)
        h_work = xvec[1]
        s_shop = xvec[2]

        c_val = c_of(h_work, s_shop)
        l_val = l_of(h_work, s_shop)

        uc_val = u_c(c_val, l_val)
        ul_val = u_l(c_val, l_val)
        p_val = p(s_shop)
        pp_val = p_prime(s_shop)

        g1 = uc_val * (w / p_val) - ul_val
        g2 = uc_val * (-(w * h_work + y) * pp_val / (p_val^2)) - ul_val

        return [g1; g2]
    end

    if !is_feasible(x0)
        return x0, 0, false
    end

    if norm(G(x0), Inf) < tol
        return x0, 0, true
    end

    for iter in 1:max_iter
        Gx = G(x)

        if !all(isfinite.(Gx))
            return x, iter, false
        end

        JG = numerical_jacobian(G, x; h=h)

        if !all(isfinite.(JG))
            return x, iter, false
        end

        if rank(JG) < length(x) || norm(JG, Inf) < eps(Float64)
            return x, iter, false
        end

        dx = JG \ Gx
        x_new = x - dx

        if !is_feasible(x_new)
            return x_new, iter, false
        end

        if norm(G(x_new), Inf) < tol
            return x_new, iter, true
        end

        x = x_new
    end

    return x, max_iter, false
end

# -------------------------------------------------------
# Construct local LQ approximation by centered differences
# Input:
#   r    : scalar return function r(x, u)
#   g    : state transition function g(x, u)
#   xbar : steady-state state vector
#   ubar : steady-state control vector
#   h    : step size for finite difference (default: 1e-6)
# Output:
#   Q, W, R, A, B for the local LQ problem
# -------------------------------------------------------
function lq_approximation(r, g, xbar, ubar; h=1e-6)
    xbar = Float64.(collect(xbar))
    ubar = Float64.(collect(ubar))

    r_xx = numerical_hessian(x -> r(x, ubar), xbar; h=h)
    r_xu = numerical_cross_hessian(r, xbar, ubar; h=h)
    r_uu = numerical_hessian(u -> r(xbar, u), ubar; h=h)

    A = numerical_jacobian(x -> g(x, ubar), xbar; h=h)
    B = numerical_jacobian(u -> g(xbar, u), ubar; h=h)

    Q = 0.5 * r_xx
    W = 0.5 * r_xu
    R = 0.5 * r_uu

    return Q, W, R, A, B
end

# -------------------------------------------------------
# Construct a local LQ approximation when the period return
# depends on individual states y, aggregate variables X3,
# and controls u, while the transition only describes the
# next-period law of motion for y.
# Input:
#   r    : scalar return function r(y, X3, u)
#   g    : transition function for y, g(y, X3, u)
#   ybar : steady-state vector of individual and exogenous states
#   x3bar: steady-state vector of aggregate variables
#   ubar : steady-state control vector
#   h    : step size for finite differences
# Output:
#   Q, W, R : local quadratic matrices in the stacked state [y; X3]
#   Ay, Az  : transition derivatives with respect to y and X3
#   By      : transition derivative with respect to u
# -------------------------------------------------------
function lq_approximation_with_aggregates(r, g, ybar, x3bar, ubar; h=1e-6)
    ybar = Float64.(collect(ybar))
    x3bar = Float64.(collect(x3bar))
    ubar = Float64.(collect(ubar))

    ny = length(ybar)
    nx3 = length(x3bar)
    xbar = [ybar; x3bar]

    r_stacked = function(x, u)
        y = x[1:ny]
        x3 = x[(ny + 1):(ny + nx3)]
        return r(y, x3, u)
    end

    g_stacked = function(x, u)
        y = x[1:ny]
        x3 = x[(ny + 1):(ny + nx3)]
        return g(y, x3, u)
    end

    r_xx = numerical_hessian(x -> r_stacked(x, ubar), xbar; h=h)
    r_xu = numerical_cross_hessian(r_stacked, xbar, ubar; h=h)
    r_uu = numerical_hessian(u -> r_stacked(xbar, u), ubar; h=h)

    Gx = numerical_jacobian(x -> g_stacked(x, ubar), xbar; h=h)
    By = numerical_jacobian(u -> g_stacked(xbar, u), ubar; h=h)

    Ay = Gx[:, 1:ny]
    Az = Gx[:, (ny + 1):(ny + nx3)]

    Q = 0.5 * r_xx
    W = 0.5 * r_xu
    R = 0.5 * r_uu

    return Q, W, R, Ay, Az, By
end

# -------------------------------------------------------
# Construct Lecture 2's distorted-equilibrium matrices from
# the local quadratic approximation and market-clearing maps
# Input:
#   Q, W, R : local quadratic matrices in stacked state [y; X3]
#   Ay, Az  : transition derivatives with respect to y and X3
#   By      : transition derivative with respect to u
#   Theta   : market-clearing matrix on y in X3 = Theta*y + Psi*u
#   Psi     : market-clearing matrix on u in X3 = Theta*y + Psi*u
#   beta    : effective discount factor after detrending
# Output:
#   NamedTuple containing Q_tilde, A_tilde_y, A_tilde_z,
#   B_tilde_y, Theta_tilde, Psi_tilde, Qhat, Ahat, Bhat,
#   Abar, and the intermediate partitions Wy, Wz, Q_tilde_y,
#   and Q_tilde_z
# -------------------------------------------------------
function construct_distorted_lq_system(Q, W, R, Ay, Az, By, Theta, Psi, beta)
    Q = Float64.(Q)
    W = Float64.(W)
    R = Float64.(R)
    Ay = Float64.(Ay)
    Az = Float64.(Az)
    By = Float64.(By)
    Theta = Float64.(Theta)
    Psi = Float64.(Psi)

    ny = size(Ay, 1)
    nx3 = size(Az, 2)

    Wy = W[1:ny, :]
    Wz = W[(ny + 1):(ny + nx3), :]

    Q_tilde = Q - W * (R \ W')
    Q_tilde_y = Q_tilde[1:ny, 1:ny]
    Q_tilde_z = Q_tilde[1:ny, (ny + 1):(ny + nx3)]

    A_tilde_y = sqrt(beta) * (Ay - By * (R \ Wy'))
    A_tilde_z = sqrt(beta) * (Az - By * (R \ Wz'))
    B_tilde_y = sqrt(beta) * By

    I_x3 = Matrix{Float64}(I, size(Psi, 1), size(Psi, 1))
    temp = I_x3 + Psi * (R \ Wz')

    Theta_tilde = temp \ (Theta - Psi * (R \ Wy'))
    Psi_tilde = temp \ Psi

    Ahat = A_tilde_y + A_tilde_z * Theta_tilde
    Qhat = Q_tilde_y + Q_tilde_z * Theta_tilde
    Bhat = B_tilde_y + A_tilde_z * Psi_tilde
    Abar = A_tilde_y - B_tilde_y * (R \ (Psi_tilde' * Q_tilde_z'))

    return (
        Wy=Wy,
        Wz=Wz,
        Q_tilde=Q_tilde,
        Q_tilde_y=Q_tilde_y,
        Q_tilde_z=Q_tilde_z,
        A_tilde_y=A_tilde_y,
        A_tilde_z=A_tilde_z,
        B_tilde_y=B_tilde_y,
        Theta_tilde=Theta_tilde,
        Psi_tilde=Psi_tilde,
        Qhat=Qhat,
        Ahat=Ahat,
        Bhat=Bhat,
        Abar=Abar
    )
end

# -------------------------------------------------------
# Solve the LQ Riccati equation by fixed-point iteration
# Input:
#   Q    : return matrix on states
#   W    : return matrix on state-control cross terms
#   R    : return matrix on controls
#   A    : state transition matrix
#   B    : control transition matrix
#   beta : discount factor
#   tol  : convergence tolerance (default: 1e-10)
#   maxiter : maximum number of iterations (default: 10000)
# Output:
#   F    : optimal feedback matrix
#   P    : value function matrix
# -------------------------------------------------------
function solve_riccati(Q, W, R, A, B, beta; tol=1e-10, maxiter=10000)
    Q = Float64.(Q)
    W = Float64.(W)
    R = Float64.(R)
    A = Float64.(A)
    B = Float64.(B)

    n = size(A, 1)
    P = zeros(n, n)

    for iter in 1:maxiter
        Huu = R + beta * B' * P * B
        Hux = W' + beta * B' * P * A

        F = Huu \ Hux
        Acl = A - B * F

        P_new = Q - F' * W' - W * F + F' * R * F + beta * Acl' * P * Acl
        err = norm(P_new - P)
        P = 0.5 * (P_new + P_new')

        if err < tol
            return F, P
        end
    end

    error("Riccati iteration did not converge within maxiter.")
end

# -------------------------------------------------------
# Remove discounting and cross-product terms from a local
# LQ approximation
# Input:
#   Q, W, R, A, B : local LQ matrices with discount factor beta
#   beta          : effective discount factor
# Output:
#   Q_tilde, A_tilde, B_tilde : transformed matrices for the
#   undiscounted representation used in Lecture 2
# -------------------------------------------------------
function transform_lq_matrices(Q, W, R, A, B, beta)
    Q_tilde = Q - W * (R \ W')
    A_tilde = sqrt(beta) * (A - B * (R \ W'))
    B_tilde = sqrt(beta) * B

    return Q_tilde, A_tilde, B_tilde
end

# -------------------------------------------------------
# Solve the modified Riccati equation from Lecture 2
# Input:
#   Qhat      : modified quadratic matrix on states
#   Ahat      : modified transition matrix on states
#   Bhat      : modified transition matrix on controls
#   Abar      : modified costate transition matrix
#   Btilde_y  : partition of transformed control matrix
#   R         : quadratic matrix on controls
#   tol       : convergence tolerance
#   maxiter   : maximum number of iterations
#   verbose_every : print the sup-norm error every this many iterations
# Output:
#   F         : equilibrium feedback matrix, so u_t = -F y_t
#   P         : value-function / costate matrix
# -------------------------------------------------------
function solve_modified_riccati_equilibrium(Qhat, Ahat, Bhat, Abar, Btilde_y, R;
                                            tol=1e-10,
                                            maxiter=10000,
                                            verbose_every=50)
    Qhat = Float64.(Qhat)
    Ahat = Float64.(Ahat)
    Bhat = Float64.(Bhat)
    Abar = Float64.(Abar)
    Btilde_y = Float64.(Btilde_y)
    R = Float64.(R)

    n = size(Ahat, 1)
    P = -1.0 .* Matrix{Float64}(I, n, n)

    for iter in 1:maxiter
        middle = R + Btilde_y' * P * Bhat
        gain = middle \ (Btilde_y' * P * Ahat)

        P_new = Qhat + Abar' * P * Ahat - Abar' * P * Bhat * gain
        P_new = 0.5 * (P_new + P_new')
        err = norm(P_new - P)

        if verbose_every > 0 && iter % verbose_every == 0
            println("Modified Riccati iteration = ", iter, ", sup-norm error = ", err)
        end

        P = P_new

        if err < tol
            if verbose_every > 0 && iter % verbose_every != 0
                println("Modified Riccati iteration = ", iter, ", sup-norm error = ", err)
            end

            F = (R + Btilde_y' * P * Bhat) \ (Btilde_y' * P * Ahat)
            println("Modified Riccati iteration converged in ", iter, " iterations.")
            return F, P
        end
    end

    error("Modified Riccati iteration did not converge within maxiter.")
end

# -------------------------------------------------------
# Build the distorted Vaughan Hamiltonian matrix from
# Lecture 2
# Input:
#   Ahat, Qhat, Bhat, Abar, Btilde_y, R : distorted-equilibrium matrices
# Output:
#   H : distorted Hamiltonian matrix
# -------------------------------------------------------
function compute_distorted_vaughan_H(Ahat, Qhat, Bhat, Abar, Btilde_y, R)
    Ainv = inv(Ahat)
    M = Ainv * Bhat * (R \ Btilde_y')

    H = [Ainv         M;
         Qhat * Ainv  Qhat * M + Abar']

    return H
end

# -------------------------------------------------------
# Solve the distorted local equilibrium using Vaughan's
# method from Lecture 2
# Input:
#   Qhat, Ahat, Bhat, Abar, Btilde_y, R : distorted-equilibrium matrices
# Output:
#   F_vaughan : equilibrium feedback matrix, so u_t = -F_vaughan y_t
#   P_vaughan : value-function / costate matrix
#   H         : distorted Hamiltonian matrix
# -------------------------------------------------------
function solve_vaughan_distorted(Qhat, Ahat, Bhat, Abar, Btilde_y, R)
    H = compute_distorted_vaughan_H(Ahat, Qhat, Bhat, Abar, Btilde_y, R)

    eig = eigen(H)
    vals = eig.values
    vecs = eig.vectors

    n = size(Ahat, 1)

    idx = sortperm(abs.(vals), rev=true)
    idx_unstable = idx[1:n]

    V = vecs[:, idx_unstable]
    V11 = V[1:n, :]
    V21 = V[n+1:2n, :]

    P_vaughan = real.(V21 / V11)
    P_vaughan = 0.5 * (P_vaughan + P_vaughan')

    F_vaughan = (R + Btilde_y' * P_vaughan * Bhat) \ (Btilde_y' * P_vaughan * Ahat)
    F_vaughan = real.(F_vaughan)

    return F_vaughan, P_vaughan, H
end

# -------------------------------------------------------
# Build Vaughan's Hamiltonian matrix
# Input:
#   A_tilde, B_tilde, Q_tilde, R : standard-form LQ matrices
# Output:
#   H : Hamiltonian matrix
# -------------------------------------------------------
function compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)
    Ainv = inv(A_tilde)
    M = Ainv * B_tilde * (R \ B_tilde')
    H = [Ainv            M;
         Q_tilde * Ainv  Q_tilde * M + A_tilde']
    return H
end

# -------------------------------------------------------
# Solve the LQ problem using Vaughan's method
# Input:
#   Q, W, R, A, B : LQ matrices from the local approximation
#   beta_lq       : effective discount factor in the detrended problem
# Output:
#   F_vaughan     : feedback matrix
#   P_vaughan     : value-function matrix
#   H_vaughan     : Hamiltonian matrix
# -------------------------------------------------------
function solve_vaughan(Q, W, R, A, B, beta_lq)
    A_tilde = sqrt(beta_lq) * (A - B * (R \ W'))
    B_tilde = sqrt(beta_lq) * B
    Q_tilde = Q - W * (R \ W')

    H = compute_vaughan_H(A_tilde, B_tilde, Q_tilde, R)

    eig = eigen(H)
    vals = eig.values
    vecs = eig.vectors

    n = size(A_tilde, 1)

    # Select the eigenvectors associated with roots outside the unit circle
    idx = sortperm(abs.(vals), rev=true)
    idx_unstable = idx[1:n]

    V = vecs[:, idx_unstable]
    V11 = V[1:n, :]
    V21 = V[n+1:2n, :]

    P_vaughan = real.(V21 / V11)
    P_vaughan = (P_vaughan + P_vaughan') / 2.0

    F_tilde = (R + B_tilde' * P_vaughan * B_tilde) \ (B_tilde' * P_vaughan * A_tilde)
    F_vaughan = real.(F_tilde + R \ W')

    return F_vaughan, P_vaughan, H
end

# -------------------------------------------------------
# Standard normal cumulative distribution function
# Numerical approximation without external packages
# Input:
#   x : scalar
# Output:
#   Phi(x)
# -------------------------------------------------------
function normcdf(x)
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419

    t = 1.0 / (1.0 + p * abs(x))
    pdf = exp(-0.5 * x^2) / sqrt(2.0 * pi)

    cdf_pos = 1.0 - pdf * (b1 * t + b2 * t^2 + b3 * t^3 + b4 * t^4 + b5 * t^5)

    if x >= 0.0
        return cdf_pos
    else
        return 1.0 - cdf_pos
    end
end

# -------------------------------------------------------
# Tauchen discretization for log z_t = rho * log z_{t-1} + eps_t
# Input:
#   rho      : persistence parameter
#   sigma_e  : standard deviation of innovation
#   n_z      : number of grid points
#   m        : width parameter for Tauchen grid (default: 3.0)
# Output:
#   z_grid   : productivity grid in levels
#   Pz       : Markov transition matrix
# -------------------------------------------------------
function tauchen(rho, sigma_e, n_z; m=3.0)
    sigma_z = sigma_e / sqrt(1.0 - rho^2)
    logz_max = m * sigma_z
    logz_min = -m * sigma_z

    logz_grid = collect(range(logz_min, logz_max, length=n_z))
    step = logz_grid[2] - logz_grid[1]

    Pz = zeros(n_z, n_z)

    for i in 1:n_z
        for j in 1:n_z
            if j == 1
                cutoff = (logz_grid[1] - rho * logz_grid[i] + step / 2.0) / sigma_e
                Pz[i, j] = normcdf(cutoff)
            elseif j == n_z
                cutoff = (logz_grid[n_z] - rho * logz_grid[i] - step / 2.0) / sigma_e
                Pz[i, j] = 1.0 - normcdf(cutoff)
            else
                upper = (logz_grid[j] - rho * logz_grid[i] + step / 2.0) / sigma_e
                lower = (logz_grid[j] - rho * logz_grid[i] - step / 2.0) / sigma_e
                Pz[i, j] = normcdf(upper) - normcdf(lower)
            end
        end
    end

    z_grid = exp.(logz_grid)

    return z_grid, Pz
end

# -------------------------------------------------------
# Steady-state system for the detrended model
# Unknowns:
#   x[1] = k_ss
#   x[2] = h_ss
#   x[3] = c_ss
# Input:
#   beta_tilde : effective discount factor after detrending
#   gamma_n    : population growth rate
#   gamma_z    : technology growth rate
# Output:
#   3 equations evaluated at (k_ss, h_ss, c_ss)
# -------------------------------------------------------
function steady_state_system(x, beta_tilde, psi, theta, delta, gamma_n, gamma_z)
    k = x[1]
    h = x[2]
    c = x[3]
    G = (1.0 + gamma_n) * (1.0 + gamma_z)

    if k <= 0.0 || c <= 0.0 || h <= 0.0 || h >= 1.0
        return [NaN, NaN, NaN]
    end

    F1 = c - k^theta * h^(1.0 - theta) + (G - (1.0 - delta)) * k

    F2 = 1.0 / beta_tilde -
         (1.0 - delta + theta * k^(theta - 1.0) * h^(1.0 - theta)) / G

    F3 = ((1.0 - theta) * k^theta * h^(-theta)) / c -
         psi / (1.0 - h)

    return [F1, F2, F3]
end

# -------------------------------------------------------
# Solve the steady-state system by Newton's method
# Input:
#   beta_tilde, psi, theta, delta, gamma_n, gamma_z : model parameters
#   x0       : initial guess [k_ss, h_ss, c_ss]
#   tol      : convergence tolerance
#   max_iter : maximum number of Newton iterations
#   hstep    : step size for numerical Jacobian
# Output:
#   k_ss, h_ss, c_ss
# -------------------------------------------------------
function solve_steady_state_newton(beta_tilde, psi, theta, delta, gamma_n, gamma_z;
                                   x0=[1.0, 0.3, 0.5],
                                   tol=1e-10, max_iter=100, hstep=1e-6)

    x = Float64.(collect(x0))

    for iter in 1:max_iter
        f(xx) = steady_state_system(xx, beta_tilde, psi, theta, delta, gamma_n, gamma_z)
        F = f(x)

        if any(!isfinite, F)
            error("Newton method entered an infeasible region. Try a different initial guess.")
        end

        if maximum(abs.(F)) < tol
            println("Steady state converged in ", iter, " iterations.")
            return x[1], x[2], x[3]
        end

        J = numerical_jacobian(f, x; h=hstep)
        step = J \ F

        alpha = 1.0
        success = false

        for ls_iter in 1:20
            x_new = x - alpha * step
            k_new, h_new, c_new = x_new

            if k_new > 0.0 && c_new > 0.0 && 0.0 < h_new < 1.0
                F_new = f(x_new)
                if all(isfinite, F_new) && maximum(abs.(F_new)) < maximum(abs.(F))
                    x = x_new
                    success = true
                    break
                end
            end

            alpha *= 0.5
        end

        if !success
            error("Newton method failed to find a feasible update. Try a different initial guess.")
        end
    end

    error("Steady-state Newton solver did not converge within max_iter.")
end


# -------------------------------------------------------
# Check whether the static problem is feasible
# Since c(h) is increasing in h, it is enough to check
# whether consumption is positive near h = 1
# Input:
#   k       : current capital
#   z       : current productivity
#   kp      : next-period capital
#   theta   : capital share
#   delta   : depreciation rate
#   gamma_n : population growth rate
#   gamma_z : technology growth rate
#   hmax    : maximum labor level (default: 1.0)
# Output:
#   true if the static problem exists feasible interior solution for c* >= 0 & 0 <= h <= 1
# -------------------------------------------------------
function is_feasible_static_problem(k, z, kp, theta, delta, gamma_n, gamma_z; hmax=1.0)
    G = (1.0 + gamma_n) * (1.0 + gamma_z)
    cmax = k^theta * (z * hmax)^(1.0 - theta) - G * kp + (1.0 - delta) * k
    return cmax > 0.0
end

# -------------------------------------------------------
# Labor FOC
# Input:
#   h       : labor
#   k       : current capital
#   z       : current productivity
#   kp      : next-period capital
#   psi     : leisure weight
#   theta   : capital share
#   delta   : depreciation rate
#   gamma_n : population growth rate
#   gamma_z : technology growth rate
# Output:
#   value of the first-order condition
# -------------------------------------------------------
function labor_foc(h, k, z, kp, psi, theta, delta, gamma_n, gamma_z)
    G = (1.0 + gamma_n) * (1.0 + gamma_z)
    c = k^theta * (z * h)^(1.0 - theta) - G * kp + (1.0 - delta) * k

    if c <= 0.0 || h <= 0.0 || h >= 1.0
        return NaN
    end

    lhs = (1.0 - theta) * k^theta * z^(1.0 - theta) * h^(-theta) / c
    rhs = psi / (1.0 - h)

    return lhs - rhs
end

# -------------------------------------------------------
# Solve for h*(k, z, k') by damped Newton's method
# Input:
#   k, z, kp : current capital, productivity, next-period capital
#   psi, theta, delta, gamma_n, gamma_z : model parameters
#   h0       : initial guess for labor
#   tol      : tolerance for root-finding
#   max_iter : maximum number of Newton iterations
#   hstep    : step size for numerical derivative
#   alpha    : damping factor for Newton update
# Output:
#   h_star   : optimal labor supply
# -------------------------------------------------------
function solve_h_star(k, z, kp, psi, theta, delta, gamma_n, gamma_z;
                      h0=0.3, tol=1e-8, max_iter=100, hstep=1e-6, alpha=0.3)

    # If the static problem is infeasible, return NaN immediately
    if !is_feasible_static_problem(k, z, kp, theta, delta, gamma_n, gamma_z)
        return NaN
    end

    h = h0

    for iter in 1:max_iter
        f(hh) = labor_foc(hh, k, z, kp, psi, theta, delta, gamma_n, gamma_z)

        f_val = f(h)

        if !isfinite(f_val)
            return NaN
        end

        if abs(f_val) < tol
            break
        end

        fp_val = numerical_derivative(f, h; h=hstep)

        if !isfinite(fp_val) || abs(fp_val) < 1e-12
            return NaN
        end

        h_new = h - f_val / fp_val

        # Project the Newton update back into the feasible interval
        h_new = min(max(h_new, 1e-6), 1.0 - 1e-6)

        # Damped update
        h = alpha * h_new + (1.0 - alpha) * h
    end

    # Final feasibility and convergence check
    G = (1.0 + gamma_n) * (1.0 + gamma_z)
    c_final = k^theta * (z * h)^(1.0 - theta) - G * kp + (1.0 - delta) * k
    f_final = labor_foc(h, k, z, kp, psi, theta, delta, gamma_n, gamma_z)

    if isfinite(c_final) && c_final > 0.0 && isfinite(f_final) && abs(f_final) < tol
        return h
    else
        return NaN
    end
end

# -------------------------------------------------------
# Compute h_star on the full grid using damped Newton's method
# Input:
#   k_grid, z_grid : grids for capital and productivity
#   psi, theta, delta, gamma_n, gamma_z : model parameters
#   h0       : initial guess for labor
#   tol      : tolerance for root-finding
#   max_iter : maximum number of Newton iterations
#   hstep    : step size for numerical derivative
#   alpha    : damping factor for Newton update
# Output:
#   h_star[ik, iz, ikp] = h*(k, z, k')
# -------------------------------------------------------
function compute_h_star(k_grid, z_grid, psi, theta, delta, gamma_n, gamma_z;
                        h0=0.3, tol=1e-8, max_iter=100, hstep=1e-6, alpha=0.3)

    n_k = length(k_grid)
    n_z = length(z_grid)

    h_star = fill(NaN, n_k, n_z, n_k)

    for ik in 1:n_k
        k = k_grid[ik]

        for iz in 1:n_z
            z = z_grid[iz]

            # Use the previous successful root as the next initial guess
            h_init = h0

            for ikp in 1:n_k
                kp = k_grid[ikp]

                h_guess = solve_h_star(
                    k, z, kp, psi, theta, delta, gamma_n, gamma_z;
                    h0=h_init, tol=tol, max_iter=max_iter, hstep=hstep, alpha=alpha
                )

                h_star[ik, iz, ikp] = h_guess

                # Update the initial guess only if the current solve is successful
                if isfinite(h_guess)
                    h_init = h_guess
                end
            end
        end
    end

    return h_star
end

# -------------------------------------------------------
# Flow utility function
# Input:
#   c     : detrended consumption
#   h     : labor
#   psi   : leisure weight
#   sigma : coefficient of relative risk aversion
# Output:
#   period utility
# -------------------------------------------------------
function flow_utility(c, h, psi, sigma)
    if c <= 0.0 || h <= 0.0 || h >= 1.0
        return -Inf
    end

    return ((c * (1.0 - h)^psi)^(1.0 - sigma)) / (1.0 - sigma)
end

# -------------------------------------------------------
# Solve the detrended growth model by Howard policy iteration
# using the precomputed labor supply h_star
# State variables:
#   (k_tilde, z)
# Control variable:
#   k_tilde_next
# Labor choice:
#   recovered from h_star[ik, iz, ikp]
# Input:
#   beta_tilde, psi, sigma, gamma_n, gamma_z, theta, delta
#   k_grid, z_grid, Pz, h_star
#   tol, max_iter, howard_iter
# Output:
#   V          : value function
#   pol_kp     : policy function for next-period capital
#   pol_h      : policy function for labor
# -------------------------------------------------------
function solve_pfi_howard(beta_tilde, psi, sigma, gamma_n, gamma_z, theta, delta,
                          k_grid, z_grid, Pz, h_star;
                          tol=1e-6, max_iter=1000, howard_iter=20)

    n_k = length(k_grid)
    n_z = length(z_grid)
    G = (1.0 + gamma_n) * (1.0 + gamma_z)

    V = zeros(n_k, n_z)
    V_new = similar(V)
    reward = fill(-Inf, n_k, n_z, n_k)

    pol_kp_idx = ones(Int, n_k, n_z)
    pol_kp = zeros(n_k, n_z)
    pol_h = zeros(n_k, n_z)

    # Precompute the one-period return for each feasible (k, z, k') tuple.
    for ik in 1:n_k
        k = k_grid[ik]

        for iz in 1:n_z
            z = z_grid[iz]

            for ikp in 1:n_k
                kp = k_grid[ikp]
                h = h_star[ik, iz, ikp]

                if !isfinite(h)
                    continue
                end

                c = k^theta * (z * h)^(1.0 - theta) -
                    G * kp +
                    (1.0 - delta) * k

                u = flow_utility(c, h, psi, sigma)

                if isfinite(u)
                    reward[ik, iz, ikp] = u
                end
            end
        end
    end

    for iter in 1:max_iter
        pol_kp_idx_old = copy(pol_kp_idx)

        # Policy improvement
        for iz in 1:n_z
            for ik in 1:n_k
                best_val = -Inf
                best_kp_idx = 1

                for ikp in 1:n_k
                    u = reward[ik, iz, ikp]

                    if !isfinite(u)
                        continue
                    end

                    EV = 0.0
                    for izp in 1:n_z
                        EV += Pz[iz, izp] * V[ikp, izp]
                    end

                    val = u + beta_tilde * EV

                    if val > best_val
                        best_val = val
                        best_kp_idx = ikp
                    end
                end

                if best_val == -Inf
                    error("No feasible choice found at state (ik=$ik, iz=$iz).")
                end

                pol_kp_idx[ik, iz] = best_kp_idx
            end
        end

        n_change = count(pol_kp_idx .!= pol_kp_idx_old)
        println("Howard iteration = ", iter, ", policy changes = ", n_change)

        # Policy evaluation
        for hiter in 1:howard_iter
            for iz in 1:n_z
                for ik in 1:n_k
                    ikp = pol_kp_idx[ik, iz]
                    u = reward[ik, iz, ikp]

                    EV = 0.0
                    for izp in 1:n_z
                        EV += Pz[iz, izp] * V[ikp, izp]
                    end

                    V_new[ik, iz] = u + beta_tilde * EV
                end
            end

            V .= V_new
        end

        if iter > 1 && n_change == 0
            for iz in 1:n_z
                for ik in 1:n_k
                    ikp = pol_kp_idx[ik, iz]
                    pol_kp[ik, iz] = k_grid[ikp]
                    pol_h[ik, iz] = h_star[ik, iz, ikp]
                end
            end

            println("Howard policy iteration converged in ", iter, " iterations.")
            return V, pol_kp, pol_h
        end
    end

    error("Howard policy iteration did not converge within max_iter.")
end

# -------------------------------------------------------
# Solve the detrended growth model by standard value
# function iteration using the precomputed labor supply h_star
# State variables:
#   (k_tilde, z)
# Control variable:
#   k_tilde_next
# Labor choice:
#   recovered from h_star[ik, iz, ikp]
# Input:
#   beta_tilde, psi, sigma, gamma_n, gamma_z, theta, delta
#   k_grid, z_grid, Pz, h_star
#   tol, max_iter
# Output:
#   V          : value function
#   pol_kp     : policy function for next-period capital
#   pol_h      : policy function for labor
# -------------------------------------------------------
function solve_vfi(beta_tilde, psi, sigma, gamma_n, gamma_z, theta, delta,
                   k_grid, z_grid, Pz, h_star;
                   tol=1e-6, max_iter=1000)

    n_k = length(k_grid)
    n_z = length(z_grid)
    G = (1.0 + gamma_n) * (1.0 + gamma_z)

    V = zeros(n_k, n_z)
    V_new = similar(V)
    reward = fill(-Inf, n_k, n_z, n_k)

    pol_kp_idx = ones(Int, n_k, n_z)
    pol_kp = zeros(n_k, n_z)
    pol_h = zeros(n_k, n_z)

    # Precompute the one-period return for each feasible (k, z, k') tuple.
    for ik in 1:n_k
        k = k_grid[ik]

        for iz in 1:n_z
            z = z_grid[iz]

            for ikp in 1:n_k
                kp = k_grid[ikp]
                h = h_star[ik, iz, ikp]

                if !isfinite(h)
                    continue
                end

                c = k^theta * (z * h)^(1.0 - theta) -
                    G * kp +
                    (1.0 - delta) * k

                u = flow_utility(c, h, psi, sigma)

                if isfinite(u)
                    reward[ik, iz, ikp] = u
                end
            end
        end
    end

    for iter in 1:max_iter
        for iz in 1:n_z
            for ik in 1:n_k
                best_val = -Inf
                best_kp_idx = 1

                for ikp in 1:n_k
                    u = reward[ik, iz, ikp]

                    if !isfinite(u)
                        continue
                    end

                    EV = 0.0
                    for izp in 1:n_z
                        EV += Pz[iz, izp] * V[ikp, izp]
                    end

                    val = u + beta_tilde * EV

                    if val > best_val
                        best_val = val
                        best_kp_idx = ikp
                    end
                end

                if best_val == -Inf
                    error("No feasible choice found at state (ik=$ik, iz=$iz).")
                end

                V_new[ik, iz] = best_val
                pol_kp_idx[ik, iz] = best_kp_idx
            end
        end

        err = maximum(abs.(V_new .- V))
        if iter % 50 == 0
            println("VFI iteration = ", iter, ", sup-norm error = ", err)
        end

        V .= V_new

        if err < tol
            if iter % 50 != 0
                println("VFI iteration = ", iter, ", sup-norm error = ", err)
            end
            for iz in 1:n_z
                for ik in 1:n_k
                    ikp = pol_kp_idx[ik, iz]
                    pol_kp[ik, iz] = k_grid[ikp]
                    pol_h[ik, iz] = h_star[ik, iz, ikp]
                end
            end

            println("Value function iteration converged in ", iter, " iterations.")
            return V, pol_kp, pol_h
        end
    end

    error("Value function iteration did not converge within max_iter.")
end

# -------------------------------------------------------
# Solve the detrended growth model by LQ approximation
# and Riccati iteration
# Input:
#   beta, psi, sigma, gamma_n, gamma_z, theta, delta, rho : model parameters
#   x0_ss      : initial guess for steady state [k_ss, h_ss, c_ss]
#   tol_ss     : steady-state convergence tolerance
#   max_iter_ss: maximum iterations for steady-state solver
#   hstep_ss   : step size for steady-state Jacobian
#   hstep_lq   : step size for local LQ approximation
# Output:
#   NamedTuple containing:
#     - detrended policy functions for k', h, and c
#     - steady-state objects
#     - local LQ matrices and Riccati solution
# -------------------------------------------------------
function solve_lq_policy_functions(params;
                                   x0_ss=[1.5, 0.3, 0.4],
                                   tol_ss=1e-10,
                                   max_iter_ss=100,
                                   hstep_ss=1e-6,
                                   hstep_lq=1e-6)

    beta = params.beta
    psi = params.psi
    sigma = params.sigma
    gamma_n = params.gamma_n
    gamma_z = params.gamma_z
    theta = params.theta
    delta = params.delta
    rho = params.rho

    Gz = 1.0 + gamma_z
    G = (1.0 + gamma_n) * Gz
    beta_tilde = beta * (1.0 + gamma_n) * Gz^(1.0 - sigma)

    k_ss, h_ss, c_ss = solve_steady_state_newton(
        beta_tilde, psi, theta, delta, gamma_n, gamma_z;
        x0=x0_ss, tol=tol_ss, max_iter=max_iter_ss, hstep=hstep_ss
    )

    function r(x, u)
        k = x[1]
        a = x[2]
        kp = u[1]
        h = u[2]

        c = k^theta * (exp(a) * h)^(1.0 - theta) -
            G * kp +
            (1.0 - delta) * k

        return flow_utility(c, h, psi, sigma)
    end

    function g(x, u)
        a = x[2]
        kp = u[1]

        return [kp, rho * a]
    end

    xbar = [k_ss, 0.0]
    ubar = [k_ss, h_ss]

    Q, W, R, A_lin, B_lin = lq_approximation(r, g, xbar, ubar; h=hstep_lq)
    F, P = solve_riccati(Q, W, R, A_lin, B_lin, beta_tilde)

    policy_k_tilde = function(k_tilde, z)
        a = log(z)
        return ubar[1] - F[1, 1] * (k_tilde - xbar[1]) - F[1, 2] * (a - xbar[2])
    end

    policy_h_tilde = function(k_tilde, z)
        a = log(z)
        return ubar[2] - F[2, 1] * (k_tilde - xbar[1]) - F[2, 2] * (a - xbar[2])
    end

    policy_c_tilde = function(k_tilde, z)
        kp_tilde = policy_k_tilde(k_tilde, z)
        h_tilde = policy_h_tilde(k_tilde, z)

        return k_tilde^theta * (z * h_tilde)^(1.0 - theta) -
               G * kp_tilde +
               (1.0 - delta) * k_tilde
    end

    return (
        params=(;
            beta=beta,
            psi=psi,
            sigma=sigma,
            gamma_n=gamma_n,
            gamma_z=gamma_z,
            theta=theta,
            delta=delta,
            rho=rho,
            G=G,
            Gz=Gz,
            beta_tilde=beta_tilde
        ),
        steady_state=(k_ss=k_ss, h_ss=h_ss, c_ss=c_ss),
        xbar=xbar,
        ubar=ubar,
        Q=Q,
        W=W,
        R=R,
        A_lin=A_lin,
        B_lin=B_lin,
        F=F,
        P=P,
        policy_k_tilde=policy_k_tilde,
        policy_h_tilde=policy_h_tilde,
        policy_c_tilde=policy_c_tilde
    )
end

# -------------------------------------------------------
# Recover the policy functions of the original model
# from the detrended LQ policy functions
# Input:
#   theta, delta, gamma_n, gamma_z : model parameters
#   lq_solution : output from solve_lq_policy_functions
# Output:
#   NamedTuple of policy functions for the original model
#   Each policy function takes (k, z, t)
# -------------------------------------------------------
function recover_original_policy_functions_lq(params, lq_solution)
    theta = params.theta
    delta = params.delta
    gamma_n = params.gamma_n
    gamma_z = params.gamma_z

    Gz = 1.0 + gamma_z
    G = (1.0 + gamma_n) * Gz

    policy_k = function(k, z, t)
        k_tilde = k / Gz^t
        kp_tilde = lq_solution.policy_k_tilde(k_tilde, z)
        return kp_tilde * Gz^(t + 1)
    end

    policy_h = function(k, z, t)
        k_tilde = k / Gz^t
        return lq_solution.policy_h_tilde(k_tilde, z)
    end

    policy_l = function(k, z, t)
        return 1.0 - policy_h(k, z, t)
    end

    policy_c = function(k, z, t)
        k_tilde = k / Gz^t
        c_tilde = lq_solution.policy_c_tilde(k_tilde, z)
        return c_tilde * Gz^t
    end

    policy_x = function(k, z, t)
        k_tilde = k / Gz^t
        kp_tilde = lq_solution.policy_k_tilde(k_tilde, z)
        x_tilde = G * kp_tilde - (1.0 - delta) * k_tilde
        return x_tilde * Gz^t
    end

    return (
        policy_k=policy_k,
        policy_c=policy_c,
        policy_x=policy_x,
        policy_h=policy_h,
        policy_l=policy_l
    )
end

# -------------------------------------------------------
# Simulate the growth model using the detrended LQ policy
# functions and recover the original variables
# Input:
#   theta, delta, gamma_n, gamma_z, rho, sigma_e : model parameters
#   lq_solution : output from solve_lq_policy_functions
#   T              : simulation length
#   seed           : base RNG seed
#   k0          : initial capital (default: steady-state capital)
#   z0          : initial productivity level (default: 1.0)
#   n_replications : number of independent simulations to generate
#   seed_step      : increment applied to the base seed across replications
# Output:
#   If n_replications = 1, a NamedTuple containing simulated series for
#   c_t, x_t, k_t, z_t, h_t, l_t, epsilon_t, and log z_t
#   If n_replications > 1, a vector of such NamedTuples
# -------------------------------------------------------
function simulate_lq_growth_model(params, lq_solution;
                                  T=200,
                                  seed=1234,
                                  k0=nothing,
                                  z0=1.0,
                                  n0=1.0,
                                  n_replications=1,
                                  seed_step=1)

    if n_replications < 1
        error("n_replications must be at least 1.")
    end

    if n_replications > 1
        return [
            simulate_lq_growth_model(
                params,
                lq_solution;
                T=T,
                seed=seed + seed_step * (rep - 1),
                k0=k0,
                z0=z0,
                n0=n0,
                n_replications=1,
                seed_step=seed_step
            )
            for rep in 1:n_replications
        ]
    end

    theta = params.theta
    delta = params.delta
    gamma_n = params.gamma_n
    gamma_z = params.gamma_z
    rho = params.rho
    sigma_e = params.sigma_e

    rng = MersenneTwister(seed)

    Gn = 1.0 + gamma_n
    Gz = 1.0 + gamma_z
    G = Gn * Gz

    if k0 === nothing
        k0 = lq_solution.steady_state.k_ss
    end

    epsilon = sigma_e .* randn(rng, T)

    n = zeros(T + 1)
    k = zeros(T + 1)
    z = zeros(T + 1)
    logz = zeros(T + 1)

    c = zeros(T)
    x = zeros(T)
    h = zeros(T)
    l = zeros(T)

    n[1] = n0
    k[1] = k0
    z[1] = z0
    logz[1] = log(z0)

    for t in 0:(T - 1)
        idx = t + 1

        k_t = k[idx]
        z_t = z[idx]
        k_tilde = k_t / Gz^t

        kp_tilde = lq_solution.policy_k_tilde(k_tilde, z_t)
        h_t = lq_solution.policy_h_tilde(k_tilde, z_t)
        c_tilde = k_tilde^theta * (z_t * h_t)^(1.0 - theta) -
                  G * kp_tilde +
                  (1.0 - delta) * k_tilde
        x_tilde = G * kp_tilde - (1.0 - delta) * k_tilde

        if kp_tilde < 0.0
            error("Simulated detrended next-period capital is negative at t=$t.")
        end

        if h_t <= 0.0 || h_t >= 1.0
            error("Simulated labor is outside (0,1) at t=$t.")
        end

        if c_tilde <= 0.0
            error("Simulated detrended consumption is nonpositive at t=$t.")
        end

        c[idx] = c_tilde * Gz^t
        x[idx] = x_tilde * Gz^t
        h[idx] = h_t
        l[idx] = 1.0 - h_t

        n[idx + 1] = Gn * n[idx]
        k[idx + 1] = kp_tilde * Gz^(t + 1)
        logz[idx + 1] = rho * logz[idx] + epsilon[idx]
        z[idx + 1] = exp(logz[idx + 1])
    end

    return (
        c=c,
        x=x,
        n=n,
        k=k,
        z=z,
        h=h,
        l=l,
        epsilon=epsilon,
        logz=logz
    )
end

using XLSX
using DataFrames


# -------------------------------------------------------
# HP filter
# Input:
#   y      : vector of observations
#   lambda : smoothing parameter of the HP filter
# Output:
#   trend  : smooth trend component of y
#   cycle  : cyclical component of y, defined as y - trend
# Notes:
#   This implementation uses a dense second-difference matrix.
# -------------------------------------------------------
function hp_filter(y, lambda)
    T = length(y)

    D = zeros(T - 2, T)
    for i in 1:(T - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end

    trend = (I + lambda * (D' * D)) \ y
    cycle = y - trend

    return trend, cycle
end

# -------------------------------------------------------
# Load raw HW5 data from Excel
# Assumes:
#   row 1 = source
#   row 2 = table
#   row 3 = variable name
#   row 4 = units
#   row 5 onward = data
# Input:
#   path       : path to the Excel workbook
#   sheet_name : worksheet name
#   start_year : first year to keep
# Output:
#   df         : DataFrame with cleaned columns used in calibration
# -------------------------------------------------------
function load_raw_data(path; sheet_name="Sheet1", start_year=1948)
    xf = XLSX.readxlsx(path)
    ws = xf[sheet_name]

    raw = ws[:, :]

    source_row = vec(raw[1, :])
    table_row = vec(raw[2, :])
    var_row = vec(raw[3, :])

    function find_col(; source_contains="", table_contains="", var_contains="")
        idx = findall(1:length(var_row)) do j
            s = isnothing(source_row[j]) || ismissing(source_row[j]) ? "" : string(source_row[j])
            t = isnothing(table_row[j]) || ismissing(table_row[j]) ? "" : string(table_row[j])
            v = isnothing(var_row[j]) || ismissing(var_row[j]) ? "" : string(var_row[j])

            occursin(source_contains, s) &&
            occursin(table_contains, t) &&
            occursin(var_contains, v)
        end

        if length(idx) != 1
            error("Expected exactly one column for source='$source_contains', table='$table_contains', var='$var_contains', found $(length(idx)).")
        end

        return idx[1]
    end

    col_year = 1
    col_gdp_real = find_col(source_contains="BEA", table_contains="1.1.6", var_contains="GDP 2017 prices")
    col_employment = find_col(source_contains="BEA", table_contains="6.4A", var_contains="Employment")
    col_gdp_nominal = find_col(source_contains="BEA", table_contains="1.1.5", var_contains="GDP current prices")
    col_comp_employees = find_col(source_contains="BEA", table_contains="6.2A", var_contains="Compensation of Employees")
    col_proprietors = find_col(source_contains="BEA", table_contains="6.12A", var_contains="Proprietors' Income")
    col_taxes = find_col(source_contains="BEA", table_contains="3.5", var_contains="Taxes on Production and Imports")
    col_subsidies = find_col(source_contains="BEA", table_contains="3.13", var_contains="Subsidies")
    col_gross_investment = find_col(source_contains="BEA", table_contains="5.1", var_contains="Gross Domestic Investment")
    col_depreciation = find_col(source_contains="BEA", table_contains="5.1", var_contains="Consumption of Fixed Capital")
    col_private_assets = find_col(source_contains="BEA", table_contains="6.1", var_contains="Private fixed assets")
    col_population = find_col(source_contains="World Bank", var_contains="total")
    col_hours_employees = find_col(source_contains="BEA", table_contains="6.9B", var_contains="Hours Worked by Employees")
    col_self_employed = find_col(source_contains="BEA", table_contains="6.7B", var_contains="Self-Employed Workers")

    function to_float_col(col_idx)
        x = raw[5:end, col_idx]
        out = Vector{Union{Missing, Float64}}(undef, length(x))
        for i in eachindex(x)
            if ismissing(x[i]) || x[i] === nothing || string(x[i]) == ""
                out[i] = missing
            else
                out[i] = Float64(x[i])
            end
        end
        return out
    end

    years = Int.(raw[5:end, col_year])

    df = DataFrame(
        year = years,
        gdp_real = to_float_col(col_gdp_real),
        employment = to_float_col(col_employment),
        gdp_nominal = to_float_col(col_gdp_nominal),
        comp_employees = to_float_col(col_comp_employees),
        proprietors_income = to_float_col(col_proprietors),
        taxes_prod_imports = to_float_col(col_taxes),
        subsidies = to_float_col(col_subsidies),
        gross_investment = to_float_col(col_gross_investment),
        depreciation = to_float_col(col_depreciation),
        private_fixed_assets = to_float_col(col_private_assets),
        population = to_float_col(col_population),
        hours_employees = to_float_col(col_hours_employees),
        self_employed_workers = to_float_col(col_self_employed)
    )

    df = dropmissing(df, [:year, :gdp_real, :gdp_nominal, :employment, :comp_employees,
                           :proprietors_income, :taxes_prod_imports, :subsidies,
                           :gross_investment, :depreciation, :private_fixed_assets, :population])

    df = filter(row -> row.year >= start_year, df)

    return df
end

# -------------------------------------------------------
# Construct data moments for calibration
# Input:
#   df        : DataFrame from load_hw5_raw_data
#   hp_lambda : smoothing parameter for HP filter
# Output:
#   df2        : enriched data DataFrame
#   hours_df   : DataFrame used for hours construction
#   cycle_y_pc : HP-filtered cycle of log output per worker
#   moments    : NamedTuple of calibration data moments
# -------------------------------------------------------
function compute_data_moments(df; hp_lambda=6.25)
    df2 = copy(df)

    df2.y_per_capita_real = df2.gdp_real ./ df2.population

    population_growth_series = diff(log.(df2.population))
    output_per_worker_growth_series = diff(log.(df2.y_per_capita_real))

    factor_income = df2.gdp_nominal .-
                    df2.proprietors_income .-
                    df2.taxes_prod_imports .+
                    df2.subsidies

    df2.labor_share = df2.comp_employees ./ factor_income
    df2.capital_share = 1 .- df2.labor_share

    df2.investment_capital_ratio = df2.gross_investment ./ df2.private_fixed_assets
    df2.capital_output_ratio = df2.private_fixed_assets ./ df2.gdp_nominal
    df2.depreciation_capital_ratio = df2.depreciation ./ df2.private_fixed_assets

    hours_mask = .!ismissing.(df2.hours_employees) .& .!ismissing.(df2.self_employed_workers)
    hours_df = df2[hours_mask, :]

    hours_df.self_employed_hours =
        (hours_df.hours_employees ./ hours_df.employment) .* hours_df.self_employed_workers

    hours_df.total_hours =
        hours_df.hours_employees .+ hours_df.self_employed_hours

    hours_df.total_workers =
        hours_df.employment .+ hours_df.self_employed_workers

    hours_df.potential_hours =
        hours_df.total_workers .* 52.0 .* 100.0 ./ 1000.0

    hours_df.h = hours_df.total_hours ./ hours_df.potential_hours
    hours_df.l = 1 .- hours_df.h

    log_y_pc = log.(df2.y_per_capita_real)
    trend_y_pc, cycle_y_pc = hp_filter(log_y_pc, hp_lambda)

    output_cycle_autocorr = cor(cycle_y_pc[2:end], cycle_y_pc[1:(end - 1)])
    output_cycle_std = std(cycle_y_pc)

    moments = (
        population_growth = mean(population_growth_series),
        output_per_worker_growth = mean(output_per_worker_growth_series),
        labor_share = mean(df2.labor_share),
        capital_share = mean(df2.capital_share),
        investment_capital_ratio = mean(df2.investment_capital_ratio),
        depreciation_capital_ratio = mean(df2.depreciation_capital_ratio),
        capital_output_ratio = mean(df2.capital_output_ratio),
        average_hours = mean(hours_df.h),
        average_leisure = mean(hours_df.l),
        output_cycle_autocorr = output_cycle_autocorr,
        output_cycle_std = output_cycle_std
    )

    return df2, hours_df, cycle_y_pc, moments
end

# -------------------------------------------------------
# Compute model moments from simulated series
# Input:
#   sim           : output from simulate_lq_growth_model
#   theta         : capital share parameter
#   burn_in       : number of initial periods to discard
#   hp_lambda     : smoothing parameter for HP filter
#   sample_length : number of post-burn-in observations used for moment calculation;
#                   if omitted, use all remaining observations
# Output:
#   NamedTuple of model moments
# -------------------------------------------------------
function compute_model_moments(sim, theta; burn_in=100, hp_lambda=6.25, sample_length=nothing)
    start_idx = burn_in + 1
    last_idx = length(sim.c)

    if sample_length !== nothing
        if sample_length < 3
            error("sample_length must be at least 3.")
        end
        last_idx = burn_in + sample_length
    end

    if start_idx > length(sim.c) || last_idx > length(sim.c)
        error("Not enough simulated observations after burn-in for the requested sample_length.")
    end

    idx_flow = start_idx:last_idx
    idx_state = start_idx:last_idx

    c = sim.c[idx_flow]
    x = sim.x[idx_flow]
    h = sim.h[idx_flow]
    l = sim.l[idx_flow]
    n = sim.n[idx_state]
    k = sim.k[idx_state]

    y = c .+ x

    population_growth_model = mean(diff(log.(n)))
    output_per_worker_growth_model = mean(diff(log.(y)))

    labor_share_model = 1.0 - theta
    capital_share_model = theta
    investment_capital_ratio_model = mean(x ./ k)
    capital_output_ratio_model = mean(k ./ y)

    average_hours_model = mean(h)
    average_leisure_model = mean(l)

    log_y = log.(y)
    trend_y, cycle_y = hp_filter(log_y, hp_lambda)

    output_cycle_autocorr_model = cor(cycle_y[2:end], cycle_y[1:(end - 1)])
    output_cycle_std_model = std(cycle_y)

    return (
        population_growth = population_growth_model,
        output_per_worker_growth = output_per_worker_growth_model,
        labor_share = labor_share_model,
        capital_share = capital_share_model,
        investment_capital_ratio = investment_capital_ratio_model,
        capital_output_ratio = capital_output_ratio_model,
        average_hours = average_hours_model,
        average_leisure = average_leisure_model,
        output_cycle_autocorr = output_cycle_autocorr_model,
        output_cycle_std = output_cycle_std_model
    )
end

# -------------------------------------------------------
# Compute averaged model moments from multiple simulations
# Input:
#   sims          : vector of outputs from simulate_lq_growth_model
#   theta         : capital share parameter
#   burn_in       : number of initial periods to discard
#   hp_lambda     : smoothing parameter for HP filter
#   sample_length : number of post-burn-in observations used for moment calculation
# Output:
#   NamedTuple of averaged model moments across simulations
# -------------------------------------------------------
function compute_model_moments(sims::AbstractVector, theta; burn_in=100, hp_lambda=6.25, sample_length=nothing)
    if isempty(sims)
        error("sims must contain at least one simulation.")
    end

    rep_moments = [
        compute_model_moments(sim, theta; burn_in=burn_in, hp_lambda=hp_lambda, sample_length=sample_length)
        for sim in sims
    ]

    fields = keys(rep_moments[1])
    averaged = (; (name => mean([getproperty(m, name) for m in rep_moments]) for name in fields)...)

    return averaged
end





# -------------------------------------------------------
# Solve the deterministic steady state of the HW6 model
# by Newton's method using numerical Jacobians
#
# Unknowns:
#   x = [k_ss, h_ss]
#
# Steady-state conditions used in the solver:
#   (1) intratemporal labor FOC
#   (2) steady-state Euler equation
#
# After solving for (k_ss, h_ss), recover all remaining
# steady-state objects:
#   c_ss, x_ss, y_ss, r_ss, w_ss, kappa_ss,
#   and the full Lecture 2 state/control blocks
#   X1_ss, X2_ss, X3_ss, y_ss, u_ss, Xbar
#
# Input:
#   params   : NamedTuple containing model parameters
#              Required fields:
#                beta, psi, sigma, gamma_n, gamma_z,
#                theta, delta, P0, P
#   x0       : initial guess for [k_ss, h_ss]
#              default = [2.0, 0.30]
#   tol      : convergence tolerance for the sup-norm
#              of the residual vector
#   max_iter : maximum number of Newton iterations
#   hstep    : step size for numerical Jacobian
#   verbose  : if true, print iteration progress
#   alpha    : damping parameter for Newton updates
#
# Output:
#   NamedTuple containing:
#     converged   : convergence flag
#     iterations  : number of Newton iterations used
#     residual    : residual vector at the solution
#     X1_ss       : steady state of individual state block
#     X2_ss       : steady state of exogenous state block
#     X3_ss       : steady state of aggregate state block
#     y_ss        : stacked known-state vector [X1_ss; X2_ss]
#     u_ss        : steady-state control vector
#     Xbar        : full stacked state vector [y_ss; X3_ss]
#     k_ss        : steady-state detrended capital
#     h_ss        : steady-state labor
#     l_ss        : steady-state leisure
#     c_ss        : steady-state detrended consumption
#     x_ss        : steady-state detrended investment
#     y_output_ss : steady-state detrended output
#     r_ss        : steady-state rental rate of capital
#     w_ss        : steady-state detrended wage
#     kappa_ss    : steady-state detrended transfer
#     z_ss        : steady-state level of detrended productivity
#     g_ss        : steady-state level of detrended government spending
#     tau_c_ss    : steady-state consumption tax rate
#     tau_h_ss    : steady-state labor income tax rate
#     tau_d_ss    : steady-state distribution tax rate
#     tau_p_ss    : steady-state profit tax rate
# -------------------------------------------------------
function solve_hw6_steady_state(params;
                                x0=[2.0, 0.30],
                                tol=1e-10,
                                max_iter=200,
                                hstep=1e-6,
                                verbose=true,
                                alpha=0.5)

    beta = params.beta
    psi = params.psi
    sigma = params.sigma
    gamma_n = params.gamma_n
    gamma_z = params.gamma_z
    theta = params.theta
    delta = params.delta
    P0 = Float64.(collect(params.P0))
    P = Float64.(Matrix(params.P))

    G = (1.0 + gamma_n) * (1.0 + gamma_z)
    beta_tilde = beta * (1.0 + gamma_n) * (1.0 + gamma_z)^(1.0 - sigma)

    # Recover the deterministic steady state of the exogenous process
    # from Sbar = P0 + P * Sbar.
    nS = length(P0)
    Sbar = (Matrix{Float64}(I, nS, nS) - P) \ P0

    a_ss = Sbar[1]
    tau_c_ss = Sbar[2]
    tau_h_ss = Sbar[3]
    tau_d_ss = Sbar[4]
    tau_p_ss = Sbar[5]
    logg_ss = Sbar[6]

    z_ss = exp(a_ss)
    g_ss = exp(logg_ss)

    function ss_residuals(x)
        k = x[1]
        h = x[2]

        if k <= 0.0 || h <= 0.0 || h >= 1.0
            return [1e8, 1e8]
        end

        K = k
        H = h

        y = K^theta * (z_ss * H)^(1.0 - theta)
        r = theta * K^(theta - 1.0) * (z_ss * H)^(1.0 - theta)
        w = (1.0 - theta) * K^theta * z_ss^(1.0 - theta) * H^(-theta)

        x_ss = (G - 1.0 + delta) * k
        c_ss = y - x_ss - g_ss

        if c_ss <= 0.0
            return [1e8, 1e8]
        end

        eq1 = psi * c_ss / (1.0 - h) - (1.0 - tau_h_ss) * w / (1.0 + tau_c_ss)
        eq2 = G - beta_tilde * (1.0 + (1.0 - tau_p_ss) * (r - delta))

        return [eq1, eq2]
    end

    x = Float64.(collect(x0))

    for iter in 1:max_iter
        F = ss_residuals(x)

        if norm(F, Inf) < tol
            k_ss = x[1]
            h_ss = x[2]

            K_ss = k_ss
            H_ss = h_ss
            Y_ss = K_ss^theta * (z_ss * H_ss)^(1.0 - theta)
            r_ss = theta * K_ss^(theta - 1.0) * (z_ss * H_ss)^(1.0 - theta)
            w_ss = (1.0 - theta) * K_ss^theta * z_ss^(1.0 - theta) * H_ss^(-theta)

            X_ss = (G - 1.0 + delta) * K_ss
            C_ss = Y_ss - X_ss - g_ss
            l_ss = 1.0 - h_ss

            kappa_ss =
                tau_c_ss * C_ss +
                tau_h_ss * w_ss * H_ss +
                tau_p_ss * (r_ss * K_ss - delta * K_ss) +
                tau_d_ss * (
                    r_ss * K_ss - X_ss -
                    tau_p_ss * (r_ss * K_ss - delta * K_ss)
                ) -
                g_ss

            X1_ss = [k_ss]
            X2_ss = Sbar
            X3_ss = [K_ss, H_ss, K_ss]
            y_ss = [X1_ss; X2_ss]
            u_ss = [k_ss, h_ss]
            Xbar = [y_ss; X3_ss]

            return (
                converged=true,
                iterations=iter,
                residual=F,
                X1_ss=X1_ss,
                X2_ss=X2_ss,
                X3_ss=X3_ss,
                y_ss=y_ss,
                u_ss=u_ss,
                Xbar=Xbar,
                k_ss=k_ss,
                h_ss=h_ss,
                l_ss=l_ss,
                c_ss=C_ss,
                x_ss=X_ss,
                y_output_ss=Y_ss,
                r_ss=r_ss,
                w_ss=w_ss,
                kappa_ss=kappa_ss,
                z_ss=z_ss,
                g_ss=g_ss,
                tau_c_ss=tau_c_ss,
                tau_h_ss=tau_h_ss,
                tau_d_ss=tau_d_ss,
                tau_p_ss=tau_p_ss
            )
        end

        J = numerical_jacobian(ss_residuals, x; h=hstep)
        dx = J \ F
        x_new = x - dx

        if verbose
            println(
                "HW6 steady-state iteration = ", iter,
                ", sup-norm residual = ", norm(F, Inf),
                ", x = ", x
            )
        end

        x = alpha * x_new + (1.0 - alpha) * x
    end

    error("HW6 steady-state Newton solver did not converge within max_iter.")
end


# -------------------------------------------------------
# Construct a local quadratic approximation of a scalar
# return function r(X, u) around a steady state (Xbar, ubar)
#
# Local approximation:
#   r(X_t, u_t) ≈ X_t'QX_t + u_t'Ru_t + 2X_t'Wu_t
#
# Input:
#   r    : scalar return function r(X, u)
#          with X in R^n and u in R^m
#   Xbar : n-dimensional steady-state state vector
#   ubar : m-dimensional steady-state control vector
#   h    : finite-difference step size
#
# Output:
#   Q : n x n quadratic matrix for states
#   W : n x m cross term matrix
#   R : m x m quadratic matrix for controls
#
# Notes:
#   - Q and R are returned in the same convention used in
#     lq_approximation, namely one-half of the numerical
#     Hessians so that
#       r(X, u) ≈ X'QX + u'Ru + 2X'Wu
#   - The constant and linear terms are omitted because
#     they do not affect the LQ policy problem after
#     expanding around the steady state
# -------------------------------------------------------
function quadratic_approximation(r, Xbar, ubar; h=1e-6)
    Xbar = Float64.(collect(Xbar))
    ubar = Float64.(collect(ubar))

    r_xx = numerical_hessian(X -> r(X, ubar), Xbar; h=h)
    r_xu = numerical_cross_hessian(r, Xbar, ubar; h=h)
    r_uu = numerical_hessian(u -> r(Xbar, u), ubar; h=h)

    Q = 0.5 * r_xx
    W = 0.5 * r_xu
    R = 0.5 * r_uu

    return Q, W, R
end



end
