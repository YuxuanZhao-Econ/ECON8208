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
       solve_riccati,
       normcdf,
       tauchen,
       steady_state_system,
       solve_steady_state_newton,
       is_feasible_static_problem,
       labor_foc,
       solve_h_star,
       compute_h_star,
       flow_utility,
       solve_pfi_howard


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

    pol_kp_idx = ones(Int, n_k, n_z)
    pol_kp = zeros(n_k, n_z)
    pol_h = zeros(n_k, n_z)

    for iter in 1:max_iter
        pol_kp_idx_old = copy(pol_kp_idx)

        # Policy improvement
        for iz in 1:n_z
            z = z_grid[iz]

            for ik in 1:n_k
                k = k_grid[ik]

                best_val = -Inf
                best_kp_idx = 1

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
                z = z_grid[iz]

                for ik in 1:n_k
                    k = k_grid[ik]
                    ikp = pol_kp_idx[ik, iz]
                    kp = k_grid[ikp]
                    h = h_star[ik, iz, ikp]

                    c = k^theta * (z * h)^(1.0 - theta) -
                        G * kp +
                        (1.0 - delta) * k

                    u = flow_utility(c, h, psi, sigma)

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


end
