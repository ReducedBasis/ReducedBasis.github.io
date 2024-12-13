---
linkTitle: Documentation
title: Introduction to RBM
---

The goal of this website is to present different Reduced Basis Methods (RBM).
The target audience is people interested in the simulations of PDEs, from beginners to more seasoned persons. The list presented below is not exhaustive but it gives a good idea of how the RBM work.

See below the links for details on different RBM and simple numerical examples with jupyter notebook:

- [Click here](/docs/pod) for the link to the POD-Galerkin
- [Click here](/docs/podi) for the link to the POD+interpolation method (PODI)
- [Click here](/docs/eim) for the link to the empirical interpolation method (EIM)
- [Click here](/docs/nirb) for the link to the NIRB 2-grid method
- [Click here](/docs/pbdw) for the link to PBDW


The objective of RBM is to find very quickly accurate approximations of parameter-dependent functions of the generic form
$$\begin{equation}
  u:\Omega \times \mathcal{G} \to \mathbb{R},
\end{equation}$$
where $\Omega \in \mathbb{R}^d$ is the spatial domain ($d$ is the spatial dimension), and $\mathcal{G}\subset \mathbb{R}^{N_p}$ is the parameter domain (that can also include time), with $N_p$ the number of parameters. The function $u$ is the solution to a parameterized Partial Differential Equation (PDE) modeling a physical phenomenon. Let us denote $\mu=(\mu_1,\dots,\mu_{N_p}) \in \mathcal{G}$ the varying parameter.

 For each parameter value $\mu$, we have a new function $u(\mu) \in V$ where $V$ is a suitable Banach space and $V'$ its topological dual solving a PDE of the form
$$\begin{align}
  &\mathcal{L}(\mu)(u(\mu))=F(\mu), \textrm{ in } \Omega, \nonumber\\
  &+ \textrm{ boundary conditions on }\partial \Omega,
\end{align}$$
where $\mathcal{L}(\mu):V \to V'$ is the PDE operator that depends on $\mu \in \mathcal{G}$ and $F(\mu) \in V'$ is the PDE right-hand side which does not depend on $u$. For instance, we may consider elliptic equations of the form

$$\begin{align*}
     - \ \textrm{div} (A(\mu) \nabla u)=f(\mu) \textrm{ in } \Omega,& \nonumber \\
     u = 0 \textrm{ on } \partial \Omega. \nonumber &
\end{align*}$$

The purpose of RBM is to very quickly find a good finite dimensional approximation of any solution to the problem (2). Usually, classical methods such as Finite Volume schemes (FV) or the Finite Element Method (FEM) are used to provide an accurate approximation. This consists in solving the problem (2) in a subspace $V_h \subset V$, where $h$ is the mesh size. The obtained discrete approximations are denoted $u_h \in V_h$. RBM are not meant to replace such methods but are employed in addition to such solvers, in order to reduce the computational time.

With the RBM, we look for a solution on a manifold which implies a reduction of complexity. This complexity reduction relies on the notion of the Kolmogorov n-width. It is linked to the concept of solution manifold, which is the set of all solutions, computed with a High-Fidelity (HF) code, to the parameterized problem (2) under a parameter variation. This manifold is denoted $\mathcal{S}_h$
$$\begin{equation}
  \mathcal{S}_h=\{u_h(\mu)\in V_h| \ \mu \in \mathcal{G}\}.
\end{equation}$$
RBM can be successful if the Kolmogorov n-width is small, which means that the solution manifold  $\mathcal{S}_h$ (3) might be approximated by a finite set of well-chosen solutions.

We define the Kolmogorov n-width of $\mathcal{S}_h$ as follows:

If $\mathcal{S}_h$ is a subset of a Banach space $V$, and $\mathbf{Y}_n$ a generic n-dimensional subspace of $V$, then the deviation between $\mathcal{S}_h$ and $\mathbf{Y}_n$ is
$$\begin{equation}
    E(\mathcal{S}_h;\mathbf{Y}_n)=\underset{x\in \mathcal{S}_h}{\sup}(\underset{y \in \mathbf{Y}_n}{\inf} \lVert| {x-y}\rVert|_{V}).
\end{equation}$$
Then the Kolmogorov n-width of $\mathcal{S}_h$ in $V$ is 
$$\begin{equation}
  d_n(\mathcal{S}_h,V)= \underset{\mathbf{Y}_n}{\inf} \{ E(\mathcal{S}_h;\mathbf{Y}_n); \mathrm{ \mathbf{Y}_n \ is\ a\ \textrm{ n-dimensional}\ subspace\ of \  } V \}.
\end{equation}$$

To approximate any solution in $\mathcal{S}_h$, we create a $N$-dimensional subspace of $V_h \subset V$ denoted $V_h^N$ and a Reduced Basis (RB) of this space. RBM aim at approximating any solution belonging to $\mathcal{S}_h$ with a small number of basis functions $N$. The optimal reduced space $V_h^N$ may not be found but there exists two main algorithms to find reduced spaces that can approximate well any solution of $\mathcal{S}_h$: the Proper Orthogonal Decomposition (POD) or greedy algorithms. In general, greedy algorithms are more efficient if they are combined with aposteriori errors. In both cases, the set of basis functions is derived from HF solutions for several well chosen parameter values, $\{u_h(\mu_1),\dots,u_h(\mu_N)\}$, called snapshots. 
The small Kolmogorov n-width (5) implies that the manifold $\mathcal{S}_h$ can be approximated with very few RB functions, provided that the parameters are chosen for the RB construction properly. Thanks to that, RBM enable HF real-time simulations and widely reduce the computational costs, with speedups that can reach several orders of magnitude.


<!--more-->


## Next

{{< cards >}}
  {{< card url="/docs/pod" title="Get Started" icon="document-text" subtitle="Visit the page of the POD-Galerkin!" >}}
{{< /cards >}}
