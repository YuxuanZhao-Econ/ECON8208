module HW7Tools

using LinearAlgebra
using DataFrames
using CSV
using Plots
using Optim

export build_bca_per_capita,
       build_bca_quarterly_per_capita,
       ckm_steady_state_k_over_y,
       detrend_and_static_wedges,
       pack_bca_theta,
       unpack_bca_theta,
       bca_steady_state,
       res_wedge_residual,
       stable_root_quadratic,
       bca_state_space,
       fixexp_state_space,
       ckm_counterfactual_observations,
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
       bca_all_but_one_accounting,
       one_wedge_panel,
       all_but_one_panel,
       two_wedge_panel,
       normalize_level,
       log_deviation_from_level,
       subwindow_idx


# -------------------------------------------------------
# Build the annual per-capita BCA panel from the HW6-style cleaned frames.
# Uses the HW6 "share x real-GDP" convention to get real per-capita c, x, k,
# attaches the hours fraction h from the BEA 6.9B / 6.7B subpanel, and
# constructs the resource-constraint residual g = y - c - x.
# Input:
#   df       : DataFrame from ECON8208Tools.compute_data_moments_hw6 (wide
#              national accounts panel with gdp_real, gdp_nominal, pce,
#              gross_investment, private_fixed_assets, population, year)
#   hours_df : DataFrame with columns year, h (hours fraction)
# Output:
#   bca_data : DataFrame with columns year, y, c, x, g, h, k_data, filtered
#              to rows where every wedge input is strictly positive and
#              h in (0, 1)
# -------------------------------------------------------
function build_bca_per_capita(df, hours_df)
    hours_small = select(hours_df, :year, :h)
    df = leftjoin(df, hours_small, on = :year)
    df = filter(row -> !ismissing(row.h), df)
    df = sort(df, :year)

    years = Float64.(df.year)
    gdp_r = Float64.(df.gdp_real)
    gdp_n = Float64.(df.gdp_nominal)
    pce   = Float64.(df.pce)
    inv_n = Float64.(df.gross_investment)
    fa_n  = Float64.(df.private_fixed_assets)
    pop   = Float64.(df.population)

    y_pc = gdp_r ./ pop
    c_pc = (pce   ./ gdp_n) .* y_pc
    x_pc = (inv_n ./ gdp_n) .* y_pc
    k_pc = (fa_n  ./ gdp_n) .* y_pc
    h_pc = Float64.(df.h)
    g_pc = y_pc .- c_pc .- x_pc

    bca_data = DataFrame(
        year   = years,
        y      = y_pc,
        c      = c_pc,
        x      = x_pc,
        g      = g_pc,
        h      = h_pc,
        k_data = k_pc,
    )

    bca_data = filter(
        row -> row.y > 0 && row.c > 0 && row.x > 0 && row.g > 0 &&
               row.h > 0 && row.h < 1 && row.k_data > 0,
        bca_data,
    )

    return bca_data
end

# -------------------------------------------------------
# Model-implied steady-state capital-to-output ratio (k/y)_ss from the
# Euler equation k/y = beta_hat * theta / ((1 + tau_x)(1 - beta_hat*(1-delta))),
# with beta_hat = beta*(1+gamma_z)^(-sigma). This is the CKM pwbca.m default
# used to anchor K_0 = (k/y)_ss * y_0 in quarterly perpetual inventory.
# Input:
#   params : NamedTuple with beta, sigma, gamma_z, theta, delta
# Keyword:
#   tau_x  : steady-state investment wedge (default 0.0)
# Output:
#   scalar (k/y)_ss
# -------------------------------------------------------
function ckm_steady_state_k_over_y(params; tau_x = 0.0)
    betah = params.beta * (1.0 + params.gamma_z)^(-params.sigma)
    return betah * params.theta / ((1.0 + tau_x) * (1.0 - betah * (1.0 - params.delta)))
end

# -------------------------------------------------------
# Build the quarterly per-capita BCA panel from Yichen's cleaned CSV.
# Converts SAAR nominal series to quarterly real per-capita flows (dividing
# by 4 after deflating), normalises hours so 1948Q1 matches the annual
# ~0.25 convention, and builds the capital stock by quarterly perpetual
# inventory, anchored at K_0 = (k/y)_ss * y_0 where (k/y)_ss is the
# model-implied steady-state ratio (CKM pwbca.m convention).
# Input:
#   csv_path : path to cleaned_data_from_yichen.csv
# Keyword:
#   params   : NamedTuple of deep parameters (needed for the model-implied
#              (k/y)_ss and for the quarterly delta); required unless both
#              `k_over_y` and `delta_q` are passed explicitly
#   tau_x_ss : assumed steady-state tau_x when deriving (k/y)_ss (default 0.0)
#   k_over_y : override for the initial K/Y ratio; if omitted, computed from
#              ckm_steady_state_k_over_y(params; tau_x = tau_x_ss)
#   delta_q  : override for the quarterly depreciation rate; defaults to
#              params.delta when params is supplied
# Output:
#   bca_data_q : DataFrame with columns year (fractional Q label), y, c, x,
#                g, h, k_data; filtered for positive wedge inputs and hours
#                in (0, 1)
# -------------------------------------------------------
function build_bca_quarterly_per_capita(csv_path::AbstractString;
                                        params   = nothing,
                                        tau_x_ss = 0.0,
                                        k_over_y = nothing,
                                        delta_q  = nothing)
    raw = DataFrame(CSV.File(csv_path))

    parse_qlabel(lbl) = parse(Int, lbl[4:end]) + (parse(Int, lbl[2:2]) - 1) / 4
    years_full = parse_qlabel.(String.(raw.time_label))

    GDP_n = Float64.(raw.GDP)
    PGDP  = Float64.(raw.PGDP)
    inv_n = Float64.(raw.gross_investment) .+ Float64.(raw.durable_goods)
    gov_n = Float64.(raw.government_consumption) .+ Float64.(raw.net_exports)
    c_n   = GDP_n .- inv_n .- gov_n
    pop   = Float64.(raw.iP)

    y_saar = (GDP_n ./ PGDP) ./ pop         # SAAR real per-capita output
    y_pc   = y_saar ./ 4.0                  # convert to quarterly flow
    c_pc   = (c_n   ./ GDP_n) .* y_pc
    x_pc   = (inv_n ./ GDP_n) .* y_pc
    g_pc   = (gov_n ./ GDP_n) .* y_pc

    h_raw = Float64.(raw.total_worked_hours) ./ pop
    h_pc  = h_raw .* (0.25 / h_raw[1])

    if delta_q === nothing
        params === nothing && error("Provide `params` (preferred) or an explicit `delta_q`.")
        delta_q = params.delta
    end
    if k_over_y === nothing
        params === nothing && error("Provide `params` to derive the model-implied (k/y)_ss, or pass `k_over_y` directly.")
        k_over_y = ckm_steady_state_k_over_y(params; tau_x = tau_x_ss)
    end

    k_pc    = similar(y_pc)
    k_pc[1] = k_over_y * y_pc[1]
    for t in 1:(length(y_pc) - 1)
        k_pc[t + 1] = (1.0 - delta_q) * k_pc[t] + x_pc[t]
    end

    bca_data = DataFrame(
        year   = years_full,
        y      = y_pc,
        c      = c_pc,
        x      = x_pc,
        g      = g_pc,
        h      = h_pc,
        k_data = k_pc,
    )

    bca_data = filter(
        row -> row.y > 0 && row.c > 0 && row.x > 0 && row.g > 0 &&
               row.h > 0 && row.h < 1 && row.k_data > 0,
        bca_data,
    )

    return bca_data
end

# -------------------------------------------------------
# Detrend the per-capita BCA data and invert the static (log z, tau_h, log g)
# wedges from first-order conditions. Builds the Kalman-filter observation
# vector Y_t = (log y_tilde, log x_tilde, log h, log g_tilde)'.
# Input:
#   bca_data : DataFrame with columns year, y, c, x, g, h, k_data
#   params   : NamedTuple with beta, psi, sigma, gamma_n, gamma_z, theta, delta
# Output:
#   NamedTuple with the detrended series (y_tilde, c_tilde, x_tilde, g_tilde,
#   h_tilde, k_tilde), the static wedges (log_z, tau_h, log_g), the observables
#   matrix Yobs (T x 4), the wide DataFrame bca_static, and scalar helpers
#   (T, trend, Ggrowth).
# -------------------------------------------------------
function detrend_and_static_wedges(bca_data, params)
    T        = nrow(bca_data)
    t_index  = collect(0:(T - 1))
    trend    = (1.0 + params.gamma_z) .^ t_index
    Ggrowth  = (1.0 + params.gamma_n) * (1.0 + params.gamma_z)

    y_tilde      = bca_data.y      ./ trend
    c_tilde      = bca_data.c      ./ trend
    x_tilde      = bca_data.x      ./ trend
    g_tilde      = bca_data.g      ./ trend
    k_data_tilde = bca_data.k_data ./ trend
    h_tilde      = bca_data.h

    if any(g_tilde .<= 0)
        error("Some residual government/resource wedge observations are nonpositive; log(g_tilde) is not defined.")
    end
    log_g = log.(g_tilde)

    k_tilde = copy(k_data_tilde)

    log_z = (log.(y_tilde) .- params.theta .* log.(k_tilde) .-
             (1.0 - params.theta) .* log.(h_tilde)) ./ (1.0 - params.theta)

    tau_h = 1.0 .- (params.psi / (1.0 - params.theta)) .*
                    (c_tilde ./ y_tilde) .*
                    (h_tilde ./ (1.0 .- h_tilde))

    Yobs = hcat(log.(y_tilde), log.(x_tilde), log.(h_tilde), log_g)

    bca_static = DataFrame(
        year    = bca_data.year,
        y_tilde = y_tilde,
        c_tilde = c_tilde,
        x_tilde = x_tilde,
        h_tilde = h_tilde,
        g_tilde = g_tilde,
        k_tilde = k_tilde,
        log_z   = log_z,
        tau_h   = tau_h,
        log_g   = log_g,
    )

    return (T = T, trend = trend, Ggrowth = Ggrowth,
            y_tilde = y_tilde, c_tilde = c_tilde, x_tilde = x_tilde,
            g_tilde = g_tilde, h_tilde = h_tilde, k_tilde = k_tilde,
            log_z = log_z, tau_h = tau_h, log_g = log_g,
            Yobs = Yobs, bca_static = bca_static)
end

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
# CKM fixexp.m analogue: rebuild the BCA state-space system under the
# "fixed-expectations" counterfactual where agents' decision rule and the
# contemporaneous observation map react only to the wedges whose As-mask
# entry is 1. The chain rule on res_wedge2.m's state definition
#   log z_eff  = As(1) * log z  + (1-As(1)) * log zbar,
#   tau_l_eff  = As(2) * tau_l  + (1-As(2)) * tau_lbar,  ... etc.
# evaluated at the steady state is equivalent to elementwise-scaling the
# wedge derivatives of the Euler residual (b0, b1) by As, which is what
# we do here. The same is true for the observation map: direct wedge
# entries in row_y, row_h get multiplied by As(1), As(2), As(4); row_x has
# no direct wedge columns; row_g is [0,0,0,0,As(4)].
# Input:
#   theta_mle : 30-dim parameter vector (used only via unpack_bca_theta /
#               bca_steady_state)
#   params    : NamedTuple of deep parameters
#   As        : 4-vector mask for (log z, tau_h, tau_x, log g)
# Output:
#   NamedTuple with the same fields as bca_state_space (Sbar, P0, P, Q,
#   A, B, C, X0, Y0, gamma, steady_state) but with all As-gated quantities
#   applied (the steady state itself is unchanged).
# -------------------------------------------------------
function fixexp_state_space(theta_mle, params, As)
    @assert length(As) == 4 "As must be a 4-vector for (log z, tau_h, tau_x, log g)."
    As_v = Float64.(collect(As))

    Sbar, P0, P, Q = unpack_bca_theta(theta_mle)
    ss = bca_steady_state(Sbar, params)
    k = ss.k; x = ss.x; y = ss.y; l = ss.h
    z = ss.z; taul = ss.tau_h; taux = ss.tau_x; g = ss.g

    psi = params.psi
    gn = params.gamma_n; gz = params.gamma_z
    theta = params.theta; delta = params.delta

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
    b0_raw = dR[[4, 6, 8, 10]]
    b1_raw = dR[[5, 7, 9, 11]]
    b0 = b0_raw .* As_v
    b1 = b1_raw .* As_v
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

    row_y = [phiyk, phiyz * As_v[1], phiyl * As_v[2], 0.0, phiyg * As_v[4]]' .+ phiykp .* gamma[1:5]'
    row_x = [phixk, 0.0,              0.0,             0.0, 0.0            ]' .+ phixkp .* gamma[1:5]'
    row_h = [philk, philz * As_v[1], phill * As_v[2], 0.0, philg * As_v[4]]' .+ philkp .* gamma[1:5]'
    row_g = [0.0 0.0 0.0 0.0 As_v[4]]
    C5 = [row_y; row_x; row_h; row_g]

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
# CKM pwbca.m-style counterfactual observation paths.
# Mirrors pwbca.m lines 105-123. Builds the "observation matrix" Xt_data of
# shape T x 6 from the data-implied log-linear wedge paths
#   Xt_data = [lkt  lzt  tault  tauxt  lgt  ones(T,1)],
# builds five C-matrices via fixexp_state_space:
#   C0 = As = [0,0,0,0]   (all wedges muted)
#   Cz = As = [1,0,0,0]   (only efficiency wedge active)
#   Cl = As = [0,1,0,0]   (only labor wedge active)
#   Cx = As = [0,0,1,0]   (only investment wedge active)
#   Cg = As = [0,0,0,1]   (only government wedge active)
# plus the full MLE C. The formula (pwbca.m lines 116-123)
#   YM_j = (Xt_data - Xt_data[ref, :]) * (C_j - C0)' + YM0[ref, :]
# with YM0 = Xt_data * C' returns the counterfactual path of
# (log y, log x, log h, log g) attributable to wedge j, rebased at the
# chosen reference period. "All-but-one" versions sum the individual wedge
# impacts minus the appropriate number of baselines (e.g. no-investment =
# C_z + C_l + C_g - 2 C0).
# Input:
#   theta_mle : 30-dim parameter vector
#   params    : NamedTuple of deep parameters
#   lkt, lzt, tault, tauxt, lgt : T-vectors of data-implied wedge paths
#   ref_index : integer row of Xt_data used as the rebase point (defaults to
#               1; pwbca.m uses Y0 = 81 which is 1979:1, the plot reference)
# Output:
#   NamedTuple with fields
#     YM_all : T x 4 model-implied observation path using the estimated C
#              (same as simulate_bca_observables with logk0 = lkt[1]; included
#              for completeness)
#     YM0    : T x 4 observation path with every wedge muted (As = 0)
#     YMz    : T x 4 observation path with only the efficiency wedge active
#     YMl    : T x 4 observation path with only the labor wedge active
#     YMx    : T x 4 observation path with only the investment wedge active
#     YMg    : T x 4 observation path with only the government wedge active
#     YMnoz  : T x 4 observation path with every wedge except z active
#     YMnol  : T x 4 observation path with every wedge except labor active
#     YMnox  : T x 4 observation path with every wedge except investment active
#     YMnog  : T x 4 observation path with every wedge except government active
#     Cmats  : NamedTuple of the raw fixexp-derived C matrices
#              (C_all, C0, Cz, Cl, Cx, Cg)
# -------------------------------------------------------
function ckm_counterfactual_observations(theta_mle, params,
                                         lkt, lzt, tault, tauxt, lgt;
                                         ref_index::Integer = 1)
    T = length(lkt)
    @assert length(lzt) == T && length(tault) == T && length(tauxt) == T && length(lgt) == T
    @assert 1 <= ref_index <= T

    ss_all = bca_state_space(theta_mle, params)
    ss_0   = fixexp_state_space(theta_mle, params, [0.0, 0.0, 0.0, 0.0])
    ss_z   = fixexp_state_space(theta_mle, params, [1.0, 0.0, 0.0, 0.0])
    ss_l   = fixexp_state_space(theta_mle, params, [0.0, 1.0, 0.0, 0.0])
    ss_x   = fixexp_state_space(theta_mle, params, [0.0, 0.0, 1.0, 0.0])
    ss_g   = fixexp_state_space(theta_mle, params, [0.0, 0.0, 0.0, 1.0])

    C_all = ss_all.C
    C0    = ss_0.C
    Cz    = ss_z.C
    Cl    = ss_l.C
    Cx    = ss_x.C
    Cg    = ss_g.C

    Xt = hcat(lkt, lzt, tault, tauxt, lgt, ones(T))
    YM0_data = Xt * C_all'

    xref = Xt[ref_index:ref_index, :]
    ymref = YM0_data[ref_index:ref_index, :]
    Xt_d = Xt .- repeat(xref, T, 1)
    ymref_rep = repeat(ymref, T, 1)

    YM_all = Xt_d * C_all' .+ ymref_rep
    YM0    = Xt_d * C0'    .+ ymref_rep
    YMz    = Xt_d * (Cz .- C0)' .+ ymref_rep
    YMl    = Xt_d * (Cl .- C0)' .+ ymref_rep
    YMx    = Xt_d * (Cx .- C0)' .+ ymref_rep
    YMg    = Xt_d * (Cg .- C0)' .+ ymref_rep
    YMnoz  = Xt_d * (Cl .+ Cx .+ Cg .- 2 .* C0)' .+ ymref_rep
    YMnol  = Xt_d * (Cz .+ Cx .+ Cg .- 2 .* C0)' .+ ymref_rep
    YMnox  = Xt_d * (Cz .+ Cl .+ Cg .- 2 .* C0)' .+ ymref_rep
    YMnog  = Xt_d * (Cz .+ Cl .+ Cx .- 2 .* C0)' .+ ymref_rep

    return (
        YM_all = YM_all, YM0 = YM0,
        YMz = YMz, YMl = YMl, YMx = YMx, YMg = YMg,
        YMnoz = YMnoz, YMnol = YMnol, YMnox = YMnox, YMnog = YMnog,
        Cmats = (C_all = C_all, C0 = C0, Cz = Cz, Cl = Cl, Cx = Cx, Cg = Cg),
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
# CKM-style steady-state Kalman-filter log-likelihood (mleq.m lines 225-246).
# Reformulates the model
#   X_{t+1} = A X_t + B eps_{t+1},   Y_t = C X_t + omega_t,   omega_t = D omega_{t-1} + eta_t
# as the lagged-state observer
#   Ybar_t = Cbar X_t + (C B eps_{t+1} + eta_{t+1}),   Ybar_t := Y_{t+1} - D Y_t
# with Cbar = C A - D C, observation-noise covariance Rbar = R + C B B' C' and
# state-measurement cross-covariance S = B B' C'. Solves the discrete algebraic
# Riccati equation (DARE) iteratively for the steady-state one-step predictor
# covariance Sigma and gain K, then runs a single pass over Ybar using those
# fixed steady-state matrices (so Omega = Rbar + Cbar Sigma Cbar' is reused
# every period, matching CKM's kfilter).
# The first observation Y_1 is dropped so T_effective = T - 1.
# Input:
#   Y      : T x ny raw observation matrix (full sample, including Y_1)
#   A      : nx x nx state transition
#   B      : nx x neps innovation loading (B B' is the state-noise covariance)
#   C      : ny x nx observation matrix
# Keyword:
#   X0     : nx initial state (X_1 in CKM's convention), default zeros(nx)
#   D      : ny x ny measurement-noise AR(1) matrix, default zeros
#   R      : ny x ny measurement-noise covariance, default zeros
#   tol    : DARE iteration tolerance (default 1e-10)
#   maxit  : DARE iteration cap (default 2000)
# Output:
#   loglik          : Gaussian log likelihood of Ybar given X_1 = X0
#   innovations     : (T-1) x ny matrix of one-step prediction innovations
#   filtered_states : (T-1) x nx matrix of one-step predictors X_{t|t-1}
# -------------------------------------------------------
function kalman_loglik(Y, A, B, C; X0 = nothing, D = nothing, R = nothing,
                       tol = 1e-10, maxit = 2000)
    T_full, ny = size(Y)
    nx         = size(A, 1)

    x1   = X0 === nothing ? zeros(nx) : Vector{Float64}(X0)
    Dmat = D  === nothing ? zeros(ny, ny) : Matrix{Float64}(D)
    Rmat = R  === nothing ? zeros(ny, ny) : Matrix{Float64}(R)

    Q_state = B * B'
    Cbar    = C * A - Dmat * C
    Rbar    = Rmat + C * Q_state * C'
    Sxcov   = Q_state * C'

    Sigma = Matrix{Float64}(I, nx, nx) .* 10.0
    Sigma[nx, nx] = 0.0
    K_gain = zeros(nx, ny)
    Omega  = similar(Rbar)

    converged = false
    for _ in 1:maxit
        Omega_k = Cbar * Sigma * Cbar' + Rbar
        Omega_k = 0.5 .* (Omega_k + Omega_k')
        chol_k = try
            cholesky(Symmetric(Omega_k))
        catch
            return -Inf, zeros(0, ny), zeros(0, nx)
        end
        K_k      = (A * Sigma * Cbar' + Sxcov) / chol_k
        Sigma_new = A * Sigma * A' + Q_state - K_k * Omega_k * K_k'
        Sigma_new = 0.5 .* (Sigma_new + Sigma_new')
        if maximum(abs.(Sigma_new - Sigma)) < tol
            Sigma     = Sigma_new
            Omega     = Omega_k
            K_gain    = K_k
            converged = true
            break
        end
        Sigma = Sigma_new
    end
    if !converged
        Omega   = Cbar * Sigma * Cbar' + Rbar
        Omega   = 0.5 .* (Omega + Omega')
        chol_k  = try
            cholesky(Symmetric(Omega))
        catch
            return -Inf, zeros(0, ny), zeros(0, nx)
        end
        K_gain  = (A * Sigma * Cbar' + Sxcov) / chol_k
    end

    chol = try
        cholesky(Symmetric(Omega))
    catch
        return -Inf, zeros(0, ny), zeros(0, nx)
    end
    logdet_Omega = 2.0 * sum(log, diag(chol.U))

    Ybar = Y[2:T_full, :] .- Y[1:(T_full - 1), :] * Dmat'
    T    = T_full - 1

    innovations     = zeros(T, ny)
    filtered_states = zeros(T, nx)

    x_pred = x1
    loglik = 0.0
    for t in 1:T
        nu = vec(Ybar[t, :]) - Cbar * x_pred
        innovations[t, :]     .= nu
        filtered_states[t, :] .= x_pred
        loglik += -0.5 * (logdet_Omega + dot(nu, chol \ nu) + ny * log(2.0 * pi))
        x_pred = A * x_pred + K_gain * nu
    end

    return loglik, innovations, filtered_states
end

# -------------------------------------------------------
# MLE objective: build the state-space from theta, then run kalman_loglik
# and add the CKM soft stationarity penalty 500000 * max(rho(P) - 0.995, 0)^2
# scaled by 0.5 (matching mleq.m line 246). The penalty is always added, so
# the likelihood surface stays smooth and the optimizer can cross the 0.995
# boundary without discontinuity.
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
        penalty = 500000.0 * max(rho_max - 0.995, 0.0)^2

        ll, _, _ = kalman_loglik(Y, ss.A, ss.B, ss.C; X0 = ss.X0)
        return isfinite(ll) ? -ll + 0.5 * penalty : 1e12
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
function benchplots_level_contribution(Y_case, Y_no_shock, y_data0; col = 1)
    y_case     = exp.(Y_case[:, col])
    y_no_shock = exp.(Y_no_shock[:, col])
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

    y_data_level = exp.(Yobs[:, 1]);  y_data0 = y_data_level[1]
    x_data_level = exp.(Yobs[:, 2]);  x_data0 = x_data_level[1]
    h_data_level = exp.(Yobs[:, 3]);  h_data0 = h_data_level[1]

    y_all_level = exp.(Y_all[:, 1]);  y_no_level = exp.(Y_no[:, 1])
    x_all_level = exp.(Y_all[:, 2]);  x_no_level = exp.(Y_no[:, 2])
    h_all_level = exp.(Y_all[:, 3]);  h_no_level = exp.(Y_no[:, 3])

    y_z_contrib = benchplots_level_contribution(Y_z, Y_no, y_data0; col = 1)
    y_h_contrib = benchplots_level_contribution(Y_h, Y_no, y_data0; col = 1)
    y_x_contrib = benchplots_level_contribution(Y_x, Y_no, y_data0; col = 1)
    y_g_contrib = benchplots_level_contribution(Y_g, Y_no, y_data0; col = 1)

    x_z_contrib = benchplots_level_contribution(Y_z, Y_no, x_data0; col = 2)
    x_h_contrib = benchplots_level_contribution(Y_h, Y_no, x_data0; col = 2)
    x_x_contrib = benchplots_level_contribution(Y_x, Y_no, x_data0; col = 2)
    x_g_contrib = benchplots_level_contribution(Y_g, Y_no, x_data0; col = 2)

    h_z_contrib = benchplots_level_contribution(Y_z, Y_no, h_data0; col = 3)
    h_h_contrib = benchplots_level_contribution(Y_h, Y_no, h_data0; col = 3)
    h_x_contrib = benchplots_level_contribution(Y_x, Y_no, h_data0; col = 3)
    h_g_contrib = benchplots_level_contribution(Y_g, Y_no, h_data0; col = 3)

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
        x_data_level = x_data_level,
        x_all_level  = x_all_level,
        x_no_level   = x_no_level,
        x_z_contrib  = x_z_contrib,
        x_h_contrib  = x_h_contrib,
        x_x_contrib  = x_x_contrib,
        x_g_contrib  = x_g_contrib,
        h_data_level = h_data_level,
        h_all_level  = h_all_level,
        h_no_level   = h_no_level,
        h_z_contrib  = h_z_contrib,
        h_h_contrib  = h_h_contrib,
        h_x_contrib  = h_x_contrib,
        h_g_contrib  = h_g_contrib,
    )

    return counterfactual_output, cf_summary, level_series
end

# -------------------------------------------------------
# "All-but-one" accounting: for each wedge k, freeze only column k at its
# first-sample value and let the other three move. Returns level series for
# output, investment, and hours using the same benchplots contribution
# formula (difference from a no-shock baseline, re-anchored at data(1)).
# Input:
#   ss_hat, wedges, ckm_final_inputs, Yobs, years : same as bca_one_wedge_accounting
# Output:
#   level_series_abo : NamedTuple with data, all, no-shock, and four
#                      all-but-one paths for each of y, x, h. Keys follow
#                      the pattern  <var>_noz_contrib, <var>_nol_contrib,
#                      <var>_nox_contrib, <var>_nog_contrib.
# -------------------------------------------------------
function bca_all_but_one_accounting(ss_hat, wedges, ckm_final_inputs, Yobs, years)
    S_actual = hcat(wedges.log_z_linear, wedges.tau_h_linear, wedges.tau_x, wedges.log_g)
    S_first  = vec(S_actual[1, :])
    logk0    = ckm_final_inputs.lkt[1]

    Y_all, _ = simulate_bca_observables(S_actual, ss_hat; logk0 = logk0)
    Y_no,  _ = simulate_bca_observables(counterfactual_wedge_paths(Int[],       S_actual, S_first), ss_hat; logk0 = logk0)
    Y_noz, _ = simulate_bca_observables(counterfactual_wedge_paths([2, 3, 4],   S_actual, S_first), ss_hat; logk0 = logk0)
    Y_nol, _ = simulate_bca_observables(counterfactual_wedge_paths([1, 3, 4],   S_actual, S_first), ss_hat; logk0 = logk0)
    Y_nox, _ = simulate_bca_observables(counterfactual_wedge_paths([1, 2, 4],   S_actual, S_first), ss_hat; logk0 = logk0)
    Y_nog, _ = simulate_bca_observables(counterfactual_wedge_paths([1, 2, 3],   S_actual, S_first), ss_hat; logk0 = logk0)

    y_data_level = exp.(Yobs[:, 1]);  y_data0 = y_data_level[1]
    x_data_level = exp.(Yobs[:, 2]);  x_data0 = x_data_level[1]
    h_data_level = exp.(Yobs[:, 3]);  h_data0 = h_data_level[1]

    y_all_level = exp.(Y_all[:, 1]);  y_no_level = exp.(Y_no[:, 1])
    x_all_level = exp.(Y_all[:, 2]);  x_no_level = exp.(Y_no[:, 2])
    h_all_level = exp.(Y_all[:, 3]);  h_no_level = exp.(Y_no[:, 3])

    y_noz = benchplots_level_contribution(Y_noz, Y_no, y_data0; col = 1)
    y_nol = benchplots_level_contribution(Y_nol, Y_no, y_data0; col = 1)
    y_nox = benchplots_level_contribution(Y_nox, Y_no, y_data0; col = 1)
    y_nog = benchplots_level_contribution(Y_nog, Y_no, y_data0; col = 1)

    x_noz = benchplots_level_contribution(Y_noz, Y_no, x_data0; col = 2)
    x_nol = benchplots_level_contribution(Y_nol, Y_no, x_data0; col = 2)
    x_nox = benchplots_level_contribution(Y_nox, Y_no, x_data0; col = 2)
    x_nog = benchplots_level_contribution(Y_nog, Y_no, x_data0; col = 2)

    h_noz = benchplots_level_contribution(Y_noz, Y_no, h_data0; col = 3)
    h_nol = benchplots_level_contribution(Y_nol, Y_no, h_data0; col = 3)
    h_nox = benchplots_level_contribution(Y_nox, Y_no, h_data0; col = 3)
    h_nog = benchplots_level_contribution(Y_nog, Y_no, h_data0; col = 3)

    return (
        years        = years,
        y_data_level = y_data_level, y_all_level = y_all_level, y_no_level = y_no_level,
        x_data_level = x_data_level, x_all_level = x_all_level, x_no_level = x_no_level,
        h_data_level = h_data_level, h_all_level = h_all_level, h_no_level = h_no_level,
        y_noz_contrib = y_noz, y_nol_contrib = y_nol, y_nox_contrib = y_nox, y_nog_contrib = y_nog,
        x_noz_contrib = x_noz, x_nol_contrib = x_nol, x_nox_contrib = x_nox, x_nog_contrib = x_nog,
        h_noz_contrib = h_noz, h_nol_contrib = h_nol, h_nox_contrib = h_nox, h_nog_contrib = h_nog,
    )
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

# -------------------------------------------------------
# Plot helpers for the counterfactual experiments. Each builds a single
# panel (Output / Hours / Investment) with Data in black plus one or more
# model-counterfactual lines, normalised so data(1) = 100.
# -------------------------------------------------------
function one_wedge_panel(title_str, years_vec, data_level,
                         z_series, h_series, x_series, g_series;
                         show_legend = false, xlab = "", ylab = "index, first = 100")
    scale(v) = 100 .* v ./ data_level[1]
    p = plot(years_vec, scale(data_level), label = "Data",
             linewidth = 3, color = :black, legend = show_legend ? :best : false)
    plot!(p, years_vec, scale(z_series), label = "Efficiency only", linewidth = 2)
    plot!(p, years_vec, scale(h_series), label = "Labor only",      linewidth = 2, linestyle = :dash)
    plot!(p, years_vec, scale(x_series), label = "Investment only", linewidth = 2, linestyle = :dashdot)
    plot!(p, years_vec, scale(g_series), label = "Government only", linewidth = 2, linestyle = :dot)
    plot!(p, title = title_str, xlabel = xlab, ylabel = ylab)
    return p
end

function all_but_one_panel(title_str, years_vec, data_level,
                           noz_series, nol_series, nox_series, nog_series;
                           show_legend = false, xlab = "", ylab = "index, first = 100")
    scale(v) = 100 .* v ./ data_level[1]
    p = plot(years_vec, scale(data_level), label = "Data",
             linewidth = 3, color = :black, legend = show_legend ? :best : false)
    plot!(p, years_vec, scale(noz_series), label = "No efficiency", linewidth = 2)
    plot!(p, years_vec, scale(nol_series), label = "No labor",      linewidth = 2, linestyle = :dash)
    plot!(p, years_vec, scale(nox_series), label = "No investment", linewidth = 2, linestyle = :dashdot)
    plot!(p, years_vec, scale(nog_series), label = "No government", linewidth = 2, linestyle = :dot)
    plot!(p, title = title_str, xlabel = xlab, ylabel = ylab)
    return p
end

function two_wedge_panel(title_str, years_vec, data_level, subset_series;
                         subset_label = "Efficiency + labor",
                         show_legend = false, xlab = "", ylab = "index, first = 100")
    scale(v) = 100 .* v ./ data_level[1]
    p = plot(years_vec, scale(data_level), label = "Data",
             linewidth = 3, color = :black, legend = show_legend ? :best : false)
    plot!(p, years_vec, scale(subset_series), label = subset_label, linewidth = 2, linestyle = :dash)
    plot!(p, title = title_str, xlabel = xlab, ylabel = ylab)
    return p
end

end # module HW7Tools
