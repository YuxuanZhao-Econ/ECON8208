# ECON8208 Overview

This repository contains course materials, homework submissions, and presentation files for ECON 8208.

Every folder is organized with slides, Jupyter notebooks for each homework assignment. We use the Julia programming language for all code examples and homework solutions.

## Files and Folders

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





