# ECON8208 Overview

This repository contains course materials, homework submissions, and presentation files for ECON 8208.

Every folder is organized with slides, Jupyter notebooks for each homework assignment. We use the Julia programming language for all code examples and homework solutions.


## Files and Folders


### Defined Functions before

- `ECON8208Tools.jl`

  This file collects the reusable Julia functions developed across Homework 1, Homework 2, and Homework 3.

  - It includes numerical differentiation routines such as scalar derivatives, Jacobians, Hessians, and cross-partial derivatives.
  - It contains the fixed-point / Newton solvers used in Homework 1 for both scalar and vector problems.
  - It also provides the local LQ approximation routine from Homework 2 and the Riccati solver from Homework 3.
  - The file is organized as a Julia module so that the functions can be imported and reused directly in different notebooks.

### `HW1/`

This folder contains the notebook submission and the presentation slides for Homework 1.

- `HW1.ipynb`

  This notebook contains the full homework solution.

  - Part 1 builds fixed-point / root-finding routines for four cases:
    - scalar problem with known derivative
    - scalar problem with unknown derivative
    - vector problem with known Jacobian
    - vector problem with unknown Jacobian
  - Part 2 applies these routines to two economic problems:
    - the intratemporal condition in a standard growth model
    - a shopping-time model with work and shopping choices
  - The notebook also includes numerical differentiation / Jacobian routines, feasibility checks, convergence checks, and test examples.

- `HW1_ECON8208.pdf`

  This file is the slide version of Homework 1. It

  - summarizes the main idea of rewriting each economic problem as a root-finding problem.
  - explains the solver strategy used in Part 2.
  - presents the pseudocode and robustness design for Problems 2(a) and 2(b).



### `HW2/`

This folder contains the notebook submission for Homework 2.

- `HW2.ipynb`

  This notebook contains the full homework solution.

  - Problem 1 constructs the inputs for a local LQ approximation.
  - It derives and implements numerical Jacobian and second-order centered-difference routines.
  - The notebook computes the matrices `Q`, `W`, `R`, `A`, and `B` from a general return function `r(x,u)` and transition function `g(x,u)`.
  - It also includes pseudocode, derivations, and simple numerical examples to verify the implementation.


### `HW3/`

This folder contains the notebook submission and the homework prompt for Homework 3.

- `HW3.ipynb`

  This notebook contains the full homework solution.

  - It reviews the local LQ approximation from Homework 2 and uses the matrices `Q`, `W`, `R`, `A`, and `B` as inputs.
  - It derives the infinite-horizon LQ problem and the associated Riccati equation.
  - It implements a fixed-point iteration routine to solve for the feedback matrix `F` and the value-function matrix `P`.
  - It also includes test examples and an economic application based on a deterministic growth model.

### `HW4/`

This folder contains the notebook submission and the homework prompt for Homework 4.

- `HW4.ipynb`

  This notebook contains the full homework solution.

  - It applies the functions developed in Homework 1–3 to compute a linear solution for a one-sector deterministic growth model.
  - The notebook formulates the problem in terms of current capital and next-period capital, derives the steady state, and constructs the local LQ approximation.
  - It then uses the Riccati solver to obtain the linear policy rule and value-function matrix.
  - The notebook also simulates and plots the equilibrium paths for capital, consumption, and output starting from an initial capital stock below the steady state.





