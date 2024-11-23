---
linkTitle: Welcome to first version of our Documentation template!
title: Introduction
---

This website is devoted to the numerical approximation of parameter-dependent functions of the generic form
$$ u : \Omega \times \mathcal{G} \to \mathbb{R},$$

where $\Omega \in \mathbb{R}^d$ is the spatial domain ($d$ is the spatial dimension), and $\mathcal{G} \subset \mathbb{R}^{Np}$ is the parameter domain, with $Np$ the number of parameters. The function $u$ is the solution to a parameterized problem, usually a Partial Differential Equation (PDE) modeling a physical phenomenon (but it could be other kind of problems such as ODEs, $\dots$). Let us denote $\mu = (\mu_1, \dots , \mu_{Np} ) \in \mathcal{G}$ the varying parameter. For each parameter value $\mu$, we have a new function $u(\mu) \in V $ where $V$ is a suitable Banach space and $V'$ its topological dual. In case of PDEs, we consider systems of the form $$L(\mu)(u(\mu))=f(\mu), \textrm{ in } \Omega, \textrm{ with some additional aboundary conditions on }\partial \Omega, $$
where $L(\mu) : V \to V′$ is the PDE operator that depends on $\mu \in \mathcal{G}$ and $F(\mu) \in V′$ is the PDE right-hand side. For instance, we may consider elliptic equations of the form

$$- div(A(\mu) \nabla u) = f (\mu), \textrm{ in } \Omega, \textrm{ with } u = 0 \textrm{ on } \partial \Omega .$$

Reduced Basis Methods (RBM) are part of the Model Order Reduction (MOR) family. The purpose of RBM is to very quickly find a good finite dimensional approximation of any solution to the parameterized problem. Usually, classical methods such as Finite Volume schemes (FV) or the Finite Element Method (FEM) are used to provide an accurate approximation. This consists in solving the problem in a subspace $V_h \subset V$, where $h$ corresponds to the mesh size. The obtained discrete approximations are denoted $u_h \in V_h$. RBM are not meant to replace such methods but are employed in addition to such solvers, in order to reduce the computational time.

### Which tools for the simulation of parameter-dependent PDEs?
To solve a parameterized problem, a natural choice consists in seeking a solution in a Banach space with FEM for instance. This solution, which is based on the resolution of a High-Fidelity (HF) code, is costly in time. Thus, for complex applications, it is often more logical to employ RBMs.
With RBMs, we look for a solution on a manifold which implies a reduction of complexity. This complexity reduction relies on the notion of the Kolmogorov $n$-width. It is linked to the concept of solution manifold, which is the set of all solutions, computed with a HF code, to the parameterized problem under a parameter variation. This manifold is denoted $S_h$.
$S_h =\{u_h(\mu) \in V_h | \mu \in \mathcal{G}\}.$

RBM can be successful if the Kolmogorov n-width is small, which means that the solution manifold $S_h$ may be approximated by a finite set of well-chosen solutions. We define the Kolmogorov n-width of $S_h$ as follows: 

[Definition.] If $S_h$ is a subset of a Banach space $V$, and $Y_n$ a generic $n$-dimensional subspace of $V$, then the deviation between $S_h$ and $Y_n$ is:
$$E(S_h;Y_n)=\underset{x \in S_h}{sup} (\underset{y \in Y_n}{\inf} \| x-y\|_V).$$
Then the Kolmogorov $n$-width of $S_h \in V$ is
$$d_n(S_h,V)= \underset{Y_n}{\inf} \{ E(S_h;Y_n);Y_n \textrm{ is a } n \textrm{-dimentional subspace of } V \}$$

To approximate any solution in S_h, we create a $n$-dimensional subspace of $V_h \subset V$ denoted $X_h^n$ and a Reduced Basis (RB) of this space, where the basis functions are denoted $(\Phi_h^i )_{i=1,...,n}$. RBM aim at approximating any solution belonging to $S_h$ with a small number of basis functions $n$. This set of basis functions is derived from HF solutions for several well chosen parameter values, $\{u_h(\mu_1),dots,u_h(\mu_N)\}$, called snapshots.
The small Kolmogorov $n$-width implies that the manifold Sh can be approximated with very few RB functions, provided that the parameters are properly chosen for the RB construction. Thanks to that, RBM enable HF real-time simulations and widely reduce the computational costs, with speedups that can reach several orders of magnitude.






<!--more-->


## Next

{{< cards >}}
  {{< card url="getting-started" title="Get Started" icon="document-text" subtitle="Create your docs in just 5 minutes!" >}}
{{< /cards >}}
