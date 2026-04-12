# HW6

This folder contains the notebook submission, the homework prompt, and the calibration data for Homework 6.

Homework 6 studies a stochastic neoclassical growth model with distortionary fiscal wedges and government spending shocks. The model extends the Homework 5 RBC environment by introducing consumption taxes, labor income taxes, profit taxes, distribution taxes, and an exogenous government spending process.


## Files and Folders


### `HW6.ipynb`

This notebook contains the full homework solution.

- It derives the distorted household, firm, and government conditions.
- It solves for the deterministic distorted steady state.
- It constructs the local LQ approximation around the distorted steady state.
- It maps the model into the Lecture 2 distorted-LQ system using the matrices `Q`, `W`, `R`, `A_y`, `A_z`, `B_y`, `\Theta`, and `\Psi`.
- It solves the transformed system using both the modified Riccati equation and Vaughan's method.
- It recovers the policy functions in the original detrended variables.
- It simulates the calibrated model, constructs model moments, and compares them with U.S. data moments.
- It calibrates both real-side parameters and fiscal-process parameters using U.S. national accounts moments.


### `HW6.pdf`

This file is the homework prompt for Homework 6.

- It states the distorted RBC model with fiscal wedges and government spending.
- It gives the required equilibrium conditions and the notation for detrended variables.
- It asks for the Lecture 2 distorted-LQ mapping, numerical solution, simulation, and calibration exercises.


### `data/raw_data.xlsx`

This file contains the U.S. macroeconomic and fiscal data used for calibration.

- It includes the macro series already used in Homework 5, such as GDP, investment, private fixed assets, compensation of employees, population, and hours.
- It also includes BEA fiscal series used to construct effective tax-rate proxies and the government spending share.
- These data are used to build the empirical moments matched in the calibration section of `HW6.ipynb`.


### `../ECON8208Tools.jl`

Homework 6 uses additional reusable functions defined in the main course tools module.

- The module includes the distorted steady-state solver for HW6.
- It includes routines for the local quadratic approximation, the modified Riccati iteration, and Vaughan's method.
- It also includes HW6-specific data-loading, moment-construction, policy-recovery, and simulation functions.
