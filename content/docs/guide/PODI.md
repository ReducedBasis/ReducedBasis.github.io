---
title: POD-Galerkin
weight: 2
---

#
 Description

The PODI (POD+Interpolation) method is a non-intrusive version of the Galerkin-POD.

The offline part that creates the $N$ reduced basis functions remains the same. Then, a further step is added. It consists in computing the reduced coefficients for all snapshots within a training set of parameters $\mathal{G}_{Ntrain} \subset \mathcal{G}$.
We denote by $\alpha_i(\mu_k),i = 1,...,N, k = 1,...,Ntrain$ these coefficients. We obtain $Ntrain$ pairs $(\mu_k,\alpha (\mu_k))$, where $\alpha \in \mathcal{R}^N$.
Thanks to a Gaussian process regression, the function that maps the input parameters $\mu_k$ to the coefficients can be reconstructed. This function is then used during the online stage to find the interpolated new coefficients for a new given parameter $\mu \in \mathcal{G}$ and to contruct the high-dimensional solution. Different methods of interpolation might be employed. For instance, a prior sensitivity analysis of the function of interest with respect to the parameters can be done. This preprocessing phase corresponds to the active subspaces property.


# Offline

a POD procedure;

# Online 

an interpolation of the reduced coefficients

# Code:
[Code with Python - Library MORDICUS](https://gitlab.com/mor_dicus/mordicus/-/tree/master/examples/ParametrizedVariability/Regression/RegressionThermal?ref_type=heads)

# References

- [Berzins, A., Helmig, J., Key, F., & Elgeti, S. (2020). Standardized non-intrusive reduced order modeling using different regression models with application to complex flow problems. CoRR.](https://openreview.net/forum?id=RWrCa6rOUA)

-[N. Demo, M. Tezzele, and G. Rozza. (2019). A non-intrusive approach for the reconstruction of pod modal coefficients through active subspaces. Comptes Rendus Mécanique, 347(11):873– 881. Data-Based Engineering Science and Technology.](https://comptes-rendus.academie-sciences.fr/mecanique/articles/10.1016/j.crme.2019.11.012/)

- [P. G. Constantine. (2015). Active subspaces: Emerging ideas for dimension reduction in parameter studies. Society for Industrial and Applied Mathematics.](https://epubs.siam.org/doi/pdf/10.1137/1.9781611973860.bm)

- [D. Rajaram, T. G. Puranik, C. Perron, and D. N. Mavris. (2020). Non-intrusive parametric reduced order modeling using randomized algorithms. In AIAA Scitech 2020 Forum, page p. 23, 01](https://arc.aiaa.org/doi/abs/10.2514/6.2020-0417)