module HW7Tools

using LinearAlgebra
using DataFrames
import Pkg
if Base.find_package("Optim") === nothing
    Pkg.add("Optim")
end
using Optim

export pack_bca_theta,
       unpack_bca_theta,
       bca_steady_state,
       res_wedge_residual,
       stable_root_quadratic,
       bca_state_space,
       ckm_loglinear_wedge_inputs,
       kalman_loglik,
       bca_negloglik,
       estimate_bca_mle_nm,
       invert_investment_wedge,
       ckm_loglinear_reconstruction,
       counterfactual_wedge_paths,
       simulate_bca_observables,
       benchplots_level_contribution,
       bca_one_wedge_accounting,
       normalize_level,
       log_deviation_from_level,
       subwindow_idx


# -------------------------------------------------------
# Pack the wedge-process vector theta_mle = [Sbar (4); vec(P) (16); lower_triangle(Q) (10)]
# Input:
#   Sbar : 4-vector of unconditional means of (log z, tau_h, tau_x, log g)
#   P    : 4 x 4 VAR(1) transition matrix
#   Q    : 4 x 4 lower-triangular Cholesky factor of the innovation covariance
# Output:
#   theta_mle : 30-dimensional parameter vector used by the MLE
# -------------------------------------------------------
function pack_bca_theta(Sbar, P, Q)
    q_entries = Float64[]
    for j in 1:4, i in j:4
        push!(q_entries, Q[i, j])
    end
    return [collect(Sbar); vec(P); q_entries]
end

# -------------------------------------------------------
# Unpack theta_mle into its structural pieces
# Input:
#   theta_mle : 30-dim vector packed by pack_bca_theta
# Output:
#   Sbar : 4-vector of unconditional means
#   P0   : 4-vector (I - P) * Sbar, the VAR(1) intercept
#   P    : 4 x 4 VAR(1) transition matrix
#   Q    : 4 x 4 lower-triangular Cholesky factor
# -------------------------------------------------------
function unpack_bca_theta(theta_mle)
    Sbar = theta_mle[1:4]
    P    = reshape(theta_mle[5:20], 4, 4)
    Q    = zeros(4, 4)
    idx  = 21
    for j in 1:4, i in j:4
        Q[i, j] = theta_mle[idx]
        idx += 1
    end
    P0 = (Matrix{Float64}(I, 4, 4) - P) * Sbar
    return Sbar, P0, P, Q
end

# -------------------------------------------------------
# Detrended steady state (mirrors mleqtrly/mleq.m lines 141-156)
# Input:
#   Sbar   : 4-vector (log z, tau_h, tau_x, log g) of unconditional means
#   params : NamedTuple with beta, psi, sigma, gamma_n, gamma_z, theta, delta
# Output:
#   NamedTuple with detrended steady-state values
#   (k, c, x, y, h, z, tau_h, tau_x, g, kl, betah)
# -------------------------------------------------------
function bca_steady_state(Sbar, params)
    beta = params.beta; psi = params.psi; sigma = params.sigma
    gn = params.gamma_n; gz = params.gamma_z
    theta = params.theta; delta = params.delta

    z    = exp(Sbar[1])
    taul = Sbar[2]
    taux = Sbar[3]
    g    = exp(Sbar[4])

    if z <= 0 || g <= 0 || 1.0 + taux <= 0 || 1.0 - taul <= 0
        error("Infeasible steady-state wedge values in bca_steady_state.")
    end

    betah = beta * (1.0 + gz)^(-sigma)
    kl    = ((1.0 + taux) * (1.0 - betah * (1.0 - delta)) / (betah * theta))^(1.0 / (theta - 1.0)) * z
    A_ss  = (z / kl)^(1.0 - theta) - (1.0 + gz) * (1.0 + gn) + 1.0 - delta
    B_ss  = (1.0 - taul) * (1.0 - theta) * kl^theta * z^(1.0 - theta) / psi
    k     = (B_ss + g) / (A_ss + B_ss / kl)
    c     = A_ss * k - g
    l     = k / kl
    y     = k^theta * (z * l)^(1.0 - theta)
    x     = y - c - g

    if k <= 0 || c <= 0 || x <= 0 || y <= 0 || l <= 0 || l >= 1
        error("Infeasible steady state in BCA system.")
    end

    return (k = k, c = c, x = x, y = y, h = l, z = z, tau_h = taul, tau_x = taux, g = g,
            kl = kl, betah = betah)
end

# -------------------------------------------------------
# Scalar Euler residual at (k_t, k_{t+1}, k_{t+2}, S_t, S_{t+1})
# Hours h_t, h_{t+1} solved from the static labor FOC by Newton iteration
# (mirrors mleqtrly/res_wedge.m lines 37-66)
# Input:
#   Z      : 11-vector (log kpp, log kp, log k, log z1, log z,
#                       taul1, taul, taux1, taux, log g1, log g)
#   params : NamedTuple with beta, psi, sigma, gamma_n, gamma_z, theta, delta
# Output:
#   residual of the (1 + tau_x) Euler equation at Z (scalar)
# -------------------------------------------------------
function res_wedge_residual(Z, params)
    beta = params.beta; psi = params.psi; sigma = params.sigma
    gn = params.gamma_n; gz = params.gamma_z
    theta = params.theta; delta = params.delta
    betah = beta * (1.0 + gz)^(-sigma)

    kpp  = exp(Z[1])
    kp   = exp(Z[2])
    k    = exp(Z[3])
    z1   = exp(Z[4]);  z    = exp(Z[5])
    taul1 = Z[6];      taul = Z[7]
    taux1 = Z[8];      taux = Z[9]
    g1   = exp(Z[10]); g    = exp(Z[11])

    l = 0.3
    for _ in 1:50
        y  = k^theta * (z * l)^(1.0 - theta)
        c  = y - kp * (1.0 + gz) * (1.0 + gn) + (1.0 - delta) * k - g
        dy = (1.0 - theta) * y / l
        dc = dy
        f  = psi * c * l - (1.0 - taul) * (1.0 - theta) * y * (1.0 - l)
        df = psi * (dc * l + c) - (1.0 - taul) * (1.0 - theta) * (dy * (1.0 - l) - y)
        step = f / df
        l -= step
        l  = clamp(l, 1e-6, 1.0 - 1e-6)
        if abs(step) < 1e-12
            break
        end
    end
    y = k^theta * (z * l)^(1.0 - theta)
    c = y - kp * (1.0 + gz) * (1.0 + gn) + (1.0 - delta) * k - g

    l1 = 0.3
    for _ in 1:50
        y1  = kp^theta * (z1 * l1)^(1.0 - theta)
        c1  = y1 - kpp * (1.0 + gz) * (1.0 + gn) + (1.0 - delta) * kp - g1
        dy1 = (1.0 - theta) * y1 / l1
        dc1 = dy1
        f   = psi * c1 * l1 - (1.0 - taul1) * (1.0 - theta) * y1 * (1.0 - l1)
        df  = psi * (dc1 * l1 + c1) - (1.0 - taul1) * (1.0 - theta) * (dy1 * (1.0 - l1) - y1)
        step = f / df
        l1  -= step
        l1  = clamp(l1, 1e-6, 1.0 - 1e-6)
        if abs(step) < 1e-12
            break
        end
    end
    y1 = kp^theta * (z1 * l1)^(1.0 - theta)
    c1 = y1 - kpp * (1.0 + gz) * (1.0 + gn) + (1.0 - delta) * kp - g1

    return (1.0 + taux)  * c^(-sigma)  * (1.0 - l )^(psi * (1.0 - sigma)) -
           betah * c1^(-sigma) * (1.0 - l1)^(psi * (1.0 - sigma)) *
                   (theta * y1 / kp + (1.0 - delta) * (1.0 + taux1))
end

# -------------------------------------------------------
# Stable root of a gamma^2 + b gamma + c = 0 with |gamma| < 1
# Input:
#   a, b, c : scalar coefficients of the quadratic
# Output:
#   the unique real root inside the unit circle (errors if none exists)
# -------------------------------------------------------
function stable_root_quadratic(a, b, c)
    roots = if abs(a) < 1e-12
        [-c / b]
    else
        disc = complex(b^2 - 4.0 * a * c)
        [(-b + sqrt(disc)) / (2.0 * a), (-b - sqrt(disc)) / (2.0 * a)]
    end
    candidates = [real(r) for r in roots if abs(imag(r)) < 1e-8 && abs(real(r)) < 1.0]
    isempty(candidates) && error("No stable real root found for gamma_k.")
    return candidates[1]
end

# -------------------------------------------------------
# Build the full BCA state-space system
#   X_{t+1} = A X_t + B eps_{t+1},   Y_t = C X_t
# State  X_t = (log k_t, S_t, 1)' with S_t = (log z, tau_h, tau_x, log g)_t.
# Observation  Y_t = (log y, log x, log h, log g)_t.
# Uses numerical differentiation of res_wedge_residual for the log-linear
# decision rule, then assembles C via the phi-coefficients in mleq.m.
# Input:
#   theta_mle : 30-dim parameter vector packed by pack_bca_theta
#   params    : NamedTuple of deep parameters
# Output:
#   NamedTuple with fields
#     Sbar, P0, P, Q         : unpacked wedge-process pieces
#     A (6x6), B (6x4), C (4x6) : state-space matrices
#     X0 (6), Y0 (4)         : steady-state state and observation vectors
#     gamma (6)              : coefficients of the log-linear capital rule
#     steady_state           : NamedTuple returned by bca_steady_state
# -------------------------------------------------------
function bca_state_space(theta_mle, params)
    Sbar, P0, P, Q = unpack_bca_theta(theta_mle)
    ss = bca_steady_state(Sbar, params)
    k = ss.k; c = ss.c; x = ss.x; y = ss.y; l = ss.h
    z = ss.z; taul = ss.tau_h; taux = ss.tau_x; g = ss.g
    betah = ss.betah

    beta = params.beta; psi = params.psi; sigma = params.sigma
    gn = params.gamma_n; gz = params.gamma_z
    theta = params.theta; delta = params.delta

    # Numerical differentiation of the Euler residual at the steady state.
    Zss = [log(k), log(k), log(k), log(z), log(z),
           taul, taul, taux, taux, log(g), log(g)]
    dR  = zeros(11)
    for i in 1:11
        step = max(abs(Zss[i]) * 1e-5, 1e-8)
        Zp = copy(Zss); Zm = copy(Zss)
        Zp[i] += step; Zm[i] -= step
        dR[i] = (res_wedge_residual(Zp, params) - res_wedge_residual(Zm, params)) / (2.0 * step)
    end

    a0, a1, a2 = dR[1], dR[2], dR[3]
    b0 = dR[[4, 6, 8, 10]]
    b1 = dR[[5, 7, 9, 11]]
    gammak = stable_root_quadratic(a0, a1, a2)
    M      = (a0 * gammak + a1) * Matrix{Float64}(I, 4, 4) + a0 * P'
    gamma_s = -(M \ (P' * b0 + b1))
    gamma0  = (1.0 - gammak) * log(k) - dot(gamma_s, [log(z), taul, taux, log(g)])
    gamma   = [gammak; gamma_s; gamma0]

    philh  = -(psi * y * (1.0 - theta)
              + (1.0 - theta) * (1.0 - taul) * y * (1.0 - l) / l * theta
              + (1.0 - theta) * (1.0 - taul) * y)
    philk  =  (psi * y * theta + psi * (1.0 - delta) * k
              - (1.0 - theta) * (1.0 - taul) * y * (1.0 - l) / l * theta) / philh
    philz  =  (psi * y * (1.0 - theta)
              - (1.0 - theta)^2 * (1.0 - taul) * y * (1.0 - l) / l) / philh
    phill  =  ((1.0 - theta) * (1.0 - taul) * y * (1.0 - l) / l * (1.0 / (1.0 - taul))) / philh
    philg  =  (-psi * g) / philh
    philkp =  (-psi * (1.0 + gz) * (1.0 + gn) * k) / philh

    phiyk  = theta + (1.0 - theta) * philk
    phiyz  = (1.0 - theta) * (1.0 + philz)
    phiyl  = (1.0 - theta) * phill
    phiyg  = (1.0 - theta) * philg
    phiykp = (1.0 - theta) * philkp

    phixk  = -k / x * (1.0 - delta)
    phixkp =  k / x * (1.0 + gz) * (1.0 + gn)

    A = [reshape(gamma, 1, 6);
         hcat(zeros(4, 1), P, reshape(P0, 4, 1));
         0.0 0.0 0.0 0.0 0.0 1.0]
    B = [zeros(1, 4); Q; zeros(1, 4)]

    row_y = [phiyk, phiyz, phiyl, 0.0, phiyg]' .+ phiykp .* gamma[1:5]'
    row_x = [phixk, 0.0,   0.0,   0.0, 0.0  ]' .+ phixkp .* gamma[1:5]'
    row_h = [philk, philz, phill, 0.0, philg]' .+ philkp .* gamma[1:5]'
    C5 = [row_y; row_x; row_h; 0.0 0.0 0.0 0.0 1.0]

    X0 = [log(k), log(z), taul, taux, log(g), 1.0]
    Y0 = [log(y), log(x), log(l), log(g)]
    phi0 = Y0 - C5 * X0[1:5]
    C = hcat(C5, phi0)

    return (
        Sbar = Sbar, P0 = P0, P = P, Q = Q,
        A = A, B = B, C = C, X0 = X0, Y0 = Y0,
        gamma = gamma,
        steady_state = ss,
    )
end

# -------------------------------------------------------
# CKM log-linear reconstruction of capital, consumption and labor wedge
# (mirrors mleannual/wedges.m lines 77-105). Also returns the nonlinear
# level z and labor wedge for comparison.
# Input:
#   ss     : NamedTuple from bca_state_space (uses ss.steady_state)
#   params : NamedTuple of deep parameters
#   y_tilde, c_tilde, x_tilde, h_tilde, g_tilde : detrended data series (T-vectors)
# Output:
#   NamedTuple with
#     lyt, lxt, llt, lgt        : log detrended observations
#     lkt, lct                  : log-linear capital and consumption paths
#     lzt                       : log-linear efficiency wedge
#     tault_linear              : log-linear labor wedge
#     K_level                   : nonlinear level capital path
#     log_z_nonlinear           : nonlinear level efficiency wedge
#     tau_h_nonlinear           : nonlinear level labor wedge
# -------------------------------------------------------
function ckm_loglinear_wedge_inputs(ss, params, y_tilde, c_tilde, x_tilde, h_tilde, g_tilde)
    T  = length(y_tilde)
    st = ss.steady_state
    Ggrowth = (1.0 + params.gamma_n) * (1.0 + params.gamma_z)

    lyt = log.(y_tilde); lxt = log.(x_tilde)
    llt = log.(h_tilde); lgt = log.(g_tilde)
    ly = log(st.y); lx = log(st.x); lc = log(st.c)
    lk = log(st.k); ll = log(st.h); lg = log(st.g); lz = log(st.z)

    lkt = similar(lyt)
    lkt[1] = lk
    for t in 1:(T - 1)
        lkt[t + 1] = lk +
            ((1.0 - params.delta) * (lkt[t] - lk) + (st.x / st.k) * (lxt[t] - lx)) / Ggrowth
    end

    lct = lc .+ (st.y .* (lyt .- ly) .- st.x .* (lxt .- lx) .- st.g .* (lgt .- lg)) ./ st.c
    lzt = lz .+ (lyt .- ly .- params.theta .* (lkt .- lk)) ./ (1.0 - params.theta) .- llt .+ ll
    tault_linear = st.tau_h .+ (1.0 - st.tau_h) .*
                  (lyt .- ly .- lct .+ lc .- (llt .- ll) ./ (1.0 - st.h))

    K_level = similar(y_tilde)
    K_level[1] = st.k
    for t in 1:(T - 1)
        K_level[t + 1] = ((1.0 - params.delta) * K_level[t] + x_tilde[t]) / Ggrowth
    end
    log_z_nonlinear = log.(
        (y_tilde ./ (K_level .^ params.theta .* h_tilde .^ (1.0 - params.theta))).^(1.0 / (1.0 - params.theta))
    )
    tau_h_nonlinear = 1.0 .- (params.psi / (1.0 - params.theta)) .*
                      (c_tilde ./ y_tilde) .* (h_tilde ./ (1.0 .- h_tilde))

    return (
        lyt = lyt, lxt = lxt, llt = llt, lgt = lgt,
        lkt = lkt, lct = lct, lzt = lzt, tault_linear = tault_linear,
        K_level = K_level,
        log_z_nonlinear = log_z_nonlinear,
        tau_h_nonlinear = tau_h_nonlinear,
    )
end

# -------------------------------------------------------
# Kalman-filter log-likelihood of Y under
#   X_{t+1} = A X_t + B eps_{t+1},   Y_t = C X_t + omega_t
# Returns -Inf if any innovation covariance fails cholesky.
# Input:
#   Y      : T x ny observation matrix
#   A      : nx x nx state transition
#   B      : nx x neps innovation loading (Q_x = B B')
#   C      : ny x nx observation matrix
# Keyword:
#   R      : ny x ny observation noise covariance (default 1e-6 I for numerical ridge)
#   X0     : nx initial filtered mean  (default zeros(nx))
#   Sigma0 : nx x nx initial filtered covariance (default 10 I)
# Output:
#   loglik          : scalar log likelihood
#   innovations     : T x ny matrix of Kalman innovations
#   filtered_states : T x nx matrix of filtered means
# -------------------------------------------------------
function kalman_loglik(Y, A, B, C; R = nothing, X0 = nothing, Sigma0 = nothing)
    T, ny = size(Y)
    nx    = size(A, 1)
    Qx    = B * B'

    x_filt     = X0     === nothing ? zeros(nx) : Vector{Float64}(X0)
    Sigma_filt = Sigma0 === nothing ? 10.0 .* Matrix{Float64}(I, nx, nx) : Matrix{Float64}(Sigma0)
    Rmat       = R      === nothing ? 1e-6 .* Matrix{Float64}(I, ny, ny) : Matrix{Float64}(R)

    loglik          = 0.0
    innovations     = zeros(T, ny)
    filtered_states = zeros(T, nx)

    for t in 1:T
        x_pred     = A * x_filt
        Sigma_pred = A * Sigma_filt * A' + Qx

        nu    = vec(Y[t, :]) - C * x_pred
        Omega = C * Sigma_pred * C' + Rmat
        Omega = 0.5 .* (Omega + Omega')

        chol = try
            cholesky(Symmetric(Omega))
        catch
            return -Inf, innovations, filtered_states
        end
        logdet_Omega = 2.0 * sum(log, diag(chol.U))
        quad         = dot(nu, chol \ nu)
        loglik      += -0.5 * (logdet_Omega + quad + ny * log(2.0 * pi))

        K_gain     = (Sigma_pred * C') / Omega
        x_filt     = x_pred + K_gain * nu
        Sigma_filt = Sigma_pred - K_gain * C * Sigma_pred
        Sigma_filt = 0.5 .* (Sigma_filt + Sigma_filt')

        innovations[t, :]     .= nu
        filtered_states[t, :] .= x_filt
    end

    return loglik, innovations, filtered_states
end

# -------------------------------------------------------
# MLE objective: build the state-space from theta, then run kalman_loglik
# Returns -L(theta) with a soft stationarity penalty on rho(P) and a large
# fallback value so a minimizer never sees NaN / Inf.
# Input:
#   theta_mle : 30-dim parameter vector
#   params    : NamedTuple of deep parameters
#   Y         : T x 4 observation matrix Yobs
# Output:
#   negative log likelihood (scalar)
# -------------------------------------------------------
function bca_negloglik(theta_mle, params, Y)
    try
        ss = bca_state_space(theta_mle, params)

        rho_max = maximum(abs.(eigvals(ss.P)))
        if rho_max >= 0.995
            return 1e10 + 1e8 * (rho_max - 0.995)^2
        end

        ll, _, _ = kalman_loglik(Y, ss.A, ss.B, ss.C; X0 = ss.X0)
        return isfinite(ll) ? -ll : 1e12
    catch
        return 1e12
    end
end

# -------------------------------------------------------
# MLE via Optim.jl Nelder-Mead with adaptive simplex restarts.
# Minimises bca_negloglik(., params, Y). Each pass rebuilds a fresh NM
# simplex around the current best; iteration stops when a pass improves
# the objective by less than improve_tol, or after max_restarts passes.
# Input:
#   theta0 : 30-dim warm-start parameter vector
#   params : NamedTuple of deep parameters
#   Y      : T x 4 observation matrix Yobs
# Keyword:
#   max_restarts : safety cap on the number of NM passes (default 10)
#   iterations   : per-pass iteration budget for Optim (default 3000)
#   improve_tol  : stop when a pass improves -L by less than this (default 1e-3)
#   f_reltol     : per-pass relative tolerance for Optim (default 1e-8)
#   verbose      : print per-pass progress (default true)
# Output:
#   theta_hat    : 30-dim minimiser of -L(theta)
#   final_neg_ll : -L(theta_hat) at the returned point
# -------------------------------------------------------
function estimate_bca_mle_nm(theta0, params, Y;
                             max_restarts::Integer = 10,
                             iterations::Integer   = 3000,
                             improve_tol::Real     = 1e-3,
                             f_reltol::Real        = 1e-8,
                             verbose::Bool         = true)
    objective = theta -> begin
        v = try
            bca_negloglik(theta, params, Y)
        catch
            1e12
        end
        isfinite(v) ? v : 1e12
    end

    nm_opts = Optim.Options(
        iterations = iterations,
        f_reltol   = f_reltol,
        show_trace = false,
    )

    theta_curr = collect(theta0)
    f_curr     = Inf
    t_total    = time()

    for k in 1:max_restarts
        t_k   = time()
        res_k = Optim.optimize(objective, theta_curr, Optim.NelderMead(), nm_opts)
        f_new = Optim.minimum(res_k)
        theta_curr = Optim.minimizer(res_k)
        improvement = isfinite(f_curr) ? f_curr - f_new : Inf

        if verbose
            println("NM restart ", k,
                    ": neg log lik = ",  f_new,
                    "  iters = ",        Optim.iterations(res_k),
                    "  improvement = ",  round(improvement; digits = 4),
                    "  elapsed = ",      round(time() - t_k; digits = 2), "s")
        end

        f_curr = f_new
        if improvement < improve_tol
            verbose && println("  -> improvement below tol = ", improve_tol, "; stopping.")
            break
        end
    end

    verbose && println("Total MLE wall time = ", round(time() - t_total; digits = 2), "s")
    return theta_curr, f_curr
end

# -------------------------------------------------------
# Invert row 2 of Y = C X for the investment wedge tau_x
# Row 2 of C reads  log x_t = C[2,1] log k_t + C[2,2] log z_t
#                             + C[2,3] tau_h_t + C[2,4] tau_x_t
#                             + C[2,5] log g_t + C[2,6]
# Input:
#   C     : 4 x 6 observation matrix from bca_state_space
#   lkt   : T-vector, log detrended capital path
#   lzt   : T-vector, log efficiency wedge
#   tault : T-vector, labor wedge
#   lgt   : T-vector, log government wedge
#   lxt   : T-vector, log detrended investment
# Output:
#   T-vector of implied investment wedge tau_x
# -------------------------------------------------------
function invert_investment_wedge(C, lkt, lzt, tault, lgt, lxt)
    return (lxt .-
            C[2, 1] .* lkt   .-
            C[2, 2] .* lzt   .-
            C[2, 3] .* tault .-
            C[2, 5] .* lgt   .-
            C[2, 6]) ./ C[2, 4]
end

# -------------------------------------------------------
# CKM-consistent wedge reconstruction after MLE (wedges.m lines 77-105)
# Builds the log-linear capital path anchored at the fitted steady state,
# the log-linear consumption and labor wedge, the nonlinear headline labor
# wedge, and finally inverts row 2 of C for tau_x.
# Input:
#   ss_hat : NamedTuple returned by bca_state_space at the MLE estimate
#   params : NamedTuple of deep parameters
#   y_tilde, x_tilde, h_tilde, g_tilde : detrended data (T-vectors)
#   log_g  : T-vector, log detrended government spending
#   years  : T-vector of year labels for the output DataFrame
# Output:
#   wedges           : DataFrame with columns year, log_z, tau_h, tau_x, log_g,
#                      log_z_linear, tau_h_linear
#   ckm_final_inputs : NamedTuple (lkt, lzt, tault_lin, lct, Ct_data) consumed
#                      by the Step-10 simulation
# -------------------------------------------------------
function ckm_loglinear_reconstruction(ss_hat, params,
                                      y_tilde, x_tilde, h_tilde, g_tilde,
                                      log_g, years)
    Ct_data = y_tilde .- x_tilde .- g_tilde
    inputs  = ckm_loglinear_wedge_inputs(ss_hat, params,
                                         y_tilde, Ct_data, x_tilde, h_tilde, g_tilde)

    tau_x = invert_investment_wedge(ss_hat.C,
                                    inputs.lkt, inputs.lzt,
                                    inputs.tault_linear, log_g, inputs.lxt)

    wedges = DataFrame(
        year         = years,
        log_z        = inputs.lzt,
        tau_h        = inputs.tau_h_nonlinear,
        tau_x        = tau_x,
        log_g        = log_g,
        log_z_linear = inputs.lzt,
        tau_h_linear = inputs.tault_linear,
    )

    ckm_final_inputs = (
        lkt       = inputs.lkt,
        lzt       = inputs.lzt,
        tault_lin = inputs.tault_linear,
        lct       = inputs.lct,
        Ct_data   = Ct_data,
    )

    return wedges, ckm_final_inputs
end

# -------------------------------------------------------
# One-wedge counterfactual paths for the setup.m / benchplots.m accounting
# Fills every column of the wedge panel with its first-sample value except
# the columns listed in active_columns, which keep their actual path.
# Input:
#   active_columns : vector of column indices in 1:4 that stay time-varying
#                    (1 = log z, 2 = tau_h, 3 = tau_x, 4 = log g)
#   actual_paths   : T x 4 matrix of measured wedge paths
#   first_state    : 4-vector, typically actual_paths[1, :]
# Output:
#   T x 4 matrix of counterfactual wedge paths
# -------------------------------------------------------
function counterfactual_wedge_paths(active_columns, actual_paths, first_state)
    T = size(actual_paths, 1)
    paths = ones(T, 1) * reshape(first_state, 1, 4)
    for j in active_columns
        paths[:, j] .= actual_paths[:, j]
    end
    return paths
end

# -------------------------------------------------------
# Forward simulation of the BCA observables given a wedge path
# Uses ss.C for the contemporaneous map and ss.gamma for the log-linear
# capital rule. Keyword logk0 fixes the initial log-capital.
# Input:
#   S_paths : T x 4 wedge path (log z, tau_h, tau_x, log g)
#   ss      : NamedTuple from bca_state_space
# Keyword:
#   logk0   : scalar initial log capital (usually ss.steady_state.k in logs)
# Output:
#   Ysim : T x 4 simulated observation matrix  (log y, log x, log h, log g)
#   Xsim : T x 6 simulated state matrix         (log k, S, 1)
# -------------------------------------------------------
function simulate_bca_observables(S_paths, ss; logk0)
    T = size(S_paths, 1)
    Xsim = zeros(T, 6)
    Ysim = zeros(T, 4)

    Xsim[1, 1] = logk0
    for t in 1:T
        Xsim[t, 2:5] .= S_paths[t, :]
        Xsim[t, 6] = 1.0
        Ysim[t, :] .= ss.C * Xsim[t, :]
        if t < T
            Xsim[t + 1, 1] = dot(ss.gamma, Xsim[t, :])
        end
    end

    return Ysim, Xsim
end

# -------------------------------------------------------
# benchplots.m-style level contribution of a one-wedge simulation
# Returns  (y_case - y_no_shock - y_case(1) + y_no_shock(1)) + y_data(1),
# a level series normalised so every case starts at the data's first value.
# Input:
#   Y_case     : T x 4 simulated observations with one wedge active
#   Y_no_shock : T x 4 simulated observations with every wedge frozen
#   y_data0    : scalar first-sample data output level
# Output:
#   T-vector level contribution of the wedge to output
# -------------------------------------------------------
function benchplots_level_contribution(Y_case, Y_no_shock, y_data0)
    y_case     = exp.(Y_case[:, 1])
    y_no_shock = exp.(Y_no_shock[:, 1])
    return (y_case .- y_no_shock .- y_case[1] .+ y_no_shock[1]) .+ y_data0
end

# -------------------------------------------------------
# Full Step-10 one-wedge accounting suite
# Runs the all-wedges, no-shock, and four one-wedge simulations, computes
# level contributions via benchplots_level_contribution, and assembles both
# the per-year table and the summary statistics (correlation, relative std)
# versus the data output series.
# Input:
#   ss_hat           : NamedTuple from bca_state_space (uses ss_hat.C and ss_hat.gamma)
#   wedges           : DataFrame returned by ckm_loglinear_reconstruction
#   ckm_final_inputs : NamedTuple returned by ckm_loglinear_reconstruction (uses .lkt[1])
#   Yobs             : T x 4 observation matrix (uses the output column)
#   years            : T-vector of year labels
# Output:
#   counterfactual_output : DataFrame with per-year indices and log deviations
#   cf_summary            : DataFrame with correlation_with_data and std_relative_to_data
#   level_series          : NamedTuple (y_data_level, y_all_level, y_no_level,
#                           y_z_contrib, y_h_contrib, y_x_contrib, y_g_contrib)
#                           used by Step-10 subwindow plots
# -------------------------------------------------------
function bca_one_wedge_accounting(ss_hat, wedges, ckm_final_inputs, Yobs, years)
    S_actual = hcat(wedges.log_z_linear, wedges.tau_h_linear, wedges.tau_x, wedges.log_g)
    S_first  = vec(S_actual[1, :])
    logk0    = ckm_final_inputs.lkt[1]

    Y_all, _ = simulate_bca_observables(S_actual, ss_hat; logk0 = logk0)
    Y_no,  _ = simulate_bca_observables(counterfactual_wedge_paths(Int[],  S_actual, S_first), ss_hat; logk0 = logk0)
    Y_z,   _ = simulate_bca_observables(counterfactual_wedge_paths([1],    S_actual, S_first), ss_hat; logk0 = logk0)
    Y_h,   _ = simulate_bca_observables(counterfactual_wedge_paths([2],    S_actual, S_first), ss_hat; logk0 = logk0)
    Y_x,   _ = simulate_bca_observables(counterfactual_wedge_paths([3],    S_actual, S_first), ss_hat; logk0 = logk0)
    Y_g,   _ = simulate_bca_observables(counterfactual_wedge_paths([4],    S_actual, S_first), ss_hat; logk0 = logk0)

    y_data_level = exp.(Yobs[:, 1])
    y_data0      = y_data_level[1]
    y_all_level  = exp.(Y_all[:, 1])
    y_no_level   = exp.(Y_no[:, 1])

    y_z_contrib = benchplots_level_contribution(Y_z, Y_no, y_data0)
    y_h_contrib = benchplots_level_contribution(Y_h, Y_no, y_data0)
    y_x_contrib = benchplots_level_contribution(Y_x, Y_no, y_data0)
    y_g_contrib = benchplots_level_contribution(Y_g, Y_no, y_data0)

    data_ld = log_deviation_from_level(y_data_level)
    all_ld  = log_deviation_from_level(y_all_level)
    no_ld   = log_deviation_from_level(y_no_level)
    z_ld    = log_deviation_from_level(y_z_contrib)
    h_ld    = log_deviation_from_level(y_h_contrib)
    x_ld    = log_deviation_from_level(y_x_contrib)
    g_ld    = log_deviation_from_level(y_g_contrib)

    counterfactual_output = DataFrame(
        year                          = years,
        data_index                    = normalize_level(y_data_level),
        all_wedges_index              = normalize_level(y_all_level),
        no_shock_index                = normalize_level(y_no_level),
        efficiency_only_index         = normalize_level(y_z_contrib),
        labor_only_index              = normalize_level(y_h_contrib),
        investment_only_index         = normalize_level(y_x_contrib),
        government_only_index         = normalize_level(y_g_contrib),
        data_log_deviation            = data_ld,
        all_wedges_log_deviation      = all_ld,
        no_shock_log_deviation        = no_ld,
        efficiency_only_log_deviation = z_ld,
        labor_only_log_deviation      = h_ld,
        investment_only_log_deviation = x_ld,
        government_only_log_deviation = g_ld,
    )

    sd(v) = sqrt(sum((v .- sum(v) / length(v)).^2) / length(v))
    cr(a, b) = begin
        ma = sum(a) / length(a); mb = sum(b) / length(b)
        da = a .- ma; db = b .- mb
        sum(da .* db) / sqrt(sum(da.^2) * sum(db.^2))
    end

    cf_summary = DataFrame(
        experiment = ["all wedges", "no shock", "efficiency only",
                      "labor only", "investment only", "government only"],
        correlation_with_data = [cr(data_ld, all_ld), cr(data_ld, no_ld),
                                 cr(data_ld, z_ld),   cr(data_ld, h_ld),
                                 cr(data_ld, x_ld),   cr(data_ld, g_ld)],
        std_relative_to_data  = [sd(all_ld) / sd(data_ld), sd(no_ld)  / sd(data_ld),
                                 sd(z_ld)   / sd(data_ld), sd(h_ld)   / sd(data_ld),
                                 sd(x_ld)   / sd(data_ld), sd(g_ld)   / sd(data_ld)],
    )

    level_series = (
        y_data_level = y_data_level,
        y_all_level  = y_all_level,
        y_no_level   = y_no_level,
        y_z_contrib  = y_z_contrib,
        y_h_contrib  = y_h_contrib,
        y_x_contrib  = y_x_contrib,
        y_g_contrib  = y_g_contrib,
    )

    return counterfactual_output, cf_summary, level_series
end

# -------------------------------------------------------
# Normalise a positive level series so the first observation equals 100
# Input:
#   level_series : T-vector of positive levels
# Output:
#   T-vector, 100 * series / series[1]
# -------------------------------------------------------
normalize_level(level_series) = 100.0 .* level_series ./ level_series[1]

# -------------------------------------------------------
# Log deviation of a positive level series from its first observation
# Input:
#   level_series : T-vector of positive levels
# Output:
#   T-vector, log(series) - log(series[1])
# -------------------------------------------------------
log_deviation_from_level(level_series) = log.(level_series) .- log(level_series[1])

# -------------------------------------------------------
# Indices of yrs falling in [t_start, t_end]
# Input:
#   yrs     : vector of year labels (may be Float64)
#   t_start : lower bound (inclusive)
#   t_end   : upper bound (inclusive)
# Output:
#   vector of indices into yrs
# -------------------------------------------------------
function subwindow_idx(yrs, t_start, t_end)
    return findall(y -> t_start <= y <= t_end, yrs)
end

end # module HW7Tools
