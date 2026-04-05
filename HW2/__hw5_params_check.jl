include(raw"C:\Users\30945\Desktop\Yuxuan ZHAO\minnesota\PhD_first_year\ECON8208\ECON8208Tools.jl")
using .ECON8208Tools
params = (
    beta = 0.96,
    psi = 1.5,
    sigma = 2.0,
    gamma_n = 0.01,
    gamma_z = 0.02,
    theta = 0.36,
    delta = 0.08,
    rho = 0.9,
    sigma_e = 0.02
)
println("loaded")
println(methods(ECON8208Tools.solve_lq_policy_functions))
println(methods(ECON8208Tools.recover_original_policy_functions_lq))
println(methods(ECON8208Tools.simulate_lq_growth_model))
