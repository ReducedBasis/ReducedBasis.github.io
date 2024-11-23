---
title: POD-Galerkin
weight: 2
---

#
 Description

Proper orthogonal decomposition (POD) is one of the most popular model reduction techniques for nonlinear partial differential equations. It is based on a Galerkin-type approximation, where the POD basis functions contain information from a solution of the dynamical system at pre-specified time instances, so-called snapshots.

# Offline

a POD procedure;

# Online 

a reduced problem to solve

# Code:
[Code with Feel++](/uploads/POD.cpp)

# References

- [Canuto, C., Tonn, T., & Urban, K. (2009). A posteriori error analysis of the reduced basis method for nonaffine parametrized nonlinear PDEs. SIAM Journal on Numerical Analysis, 47(3), 2001-2022](https://epubs.siam.org/doi/abs/10.1137/080724812)

- [Couplet, M., Basdevant, C., & Sagaut, P. (2005). Calibrated reduced-order POD-Galerkin system for fluid flow modelling. Journal of Computational Physics, 207(1), 192-220](https://www.sciencedirect.com/science/article/pii/S0021999105000239)

- [Hijazi, S., Stabile, G., Mola, A., & Rozza, G. (2020). Data-driven POD-Galerkin reduced order model for turbulent flows. Journal of Computational Physics, 416, 109513](https://www.sciencedirect.com/science/article/abs/pii/S0021999120302874)