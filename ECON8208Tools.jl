module ECON8208Tools

using LinearAlgebra

export numerical_derivative,
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
       solve_riccati

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

end
