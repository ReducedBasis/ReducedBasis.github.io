---
title: NIRB 2-grid
weight: 4
---

#### Description

Offline-Online Decomposition:
- A non-intrusive reduced basis method 

- A two-grid finite element/finite volume scheme 

- A user-friendly reduced basis method to reduce computational cost of CFD simulations.

As is well documented in the literature, the online implementation of the standard RB method (a Galerkin approach within the reduced basis space) requires to modify the original CFD calculation code, which for a commercial one may be problematic even impossible. For this reason, more  methods that are called "non-intrusive" are developed and studied than before. The NIRB 2-grid method is an alternative non-intrusive reduced basis approach (NIRB) based on a two-grid finite element discretization. Here also the process is two stages: offline, the construction of the reduced basis is performed on a fine mesh; online a new configuration is simulated using a coarse mesh. While such a coarse solution, can be computed quickly enough to be used in a rapid decision process, it is generally not accurate enough for practical use. In order to retrieve accuracy, we first project every such coarse solution into the reduced space, and then further improve them via a rectification technique. 
Instead of the usual fine coefficients, we use coarse coefficients:
$$ u_{new}=\underset{i=1}{\overset{N}{\sum}} (u_H(\mu), \Phi_i)\Phi_i(x),$$
where $(\cdot,\cdot)$ is the $L^2$ inner product, $(\Phi_i)_{i=1}^N$ corresponds to our reduced basis computed on the fine mesh, and $u_H(\mu)$ to a coarser solution generated from a new parameter of interest.

### Offline
 A Greedy or POD procedure;

### Online
Computation of a coarse solution;

+ rectification of the coefficients

## Codes:
[Python jupyter-notebook](/post/NIRB2Grid)

[Code with FreeFem++](/uploads/NIRB.edp)

## Details

The variable parameter is denoted $\mu$.
Let $\Omega$ be a bounded domain in $\mathbb{R}^d$, with $d \leq 3$ and let $u_h(\mu)$ be the solution approximation computed on a fine mesh $\mathcal{T}_h$, with a classical method, and respectively $u_H(\mu)$ be the solution approximation computed on the coarse mesh $\mathcal{T}_H$.

The fine grid is needed for the reduced basis generation, and the other one to roughly approximate the solution.  The implementation has two main steps:

   - First, the RB functions are prepared in an "offline" stage with a fine mesh. It involves a greedy algorithm or a POD procedure. This part is costly in time, but only executed once, as for other RBM.
At the end of this stage, we obtain $N$ $L^2$-orthonormalized basis functions $(\Phi_i^h)_{i=1,\dots,N}$.

   - Then, a coarse approximation of the solution, for the new parameter $\mu$ we are interested in, is computed "online".
   We denote this coarse solution $u_H(\mu)$. This rough approximation is not of sufficient precision but is calculated with a smaller number of degrees of freedom compared to the fine mesh ones.
   It is used as a cheap surrogate of the optimal coefficients
     $   \alpha_i^h(\mu)=\int_{\Omega} u_h(\mu) \cdot \Phi_i^h\ dx.$
      Reduced basis post-processing then makes it possible to notably improve the precision by projection and rectification on the reduced basis, within a very short runtime. The classical NIRB approximation is given by
$u_{Hh}^N(\mu):= \overset{N}{\underset{i=1}{\sum}}(u_H(\mu),\Phi_i^h)\ \Phi_i^h \in X_h^N.$


We enhance this approximation with a "rectification post-treatment":

The main idea of its strategy consists in recovering the accuracy of the approximation given by the optimal coefficients without sacrificing on the computational complexity. 
During the offline stage, after the fine snapshots generation, for the same parameter values, the corresponding coarse snapshots are computed.

Thus, we introduce the fine and coarse coefficients

$$\begin{equation*}
  \alpha_i^h(\mu)=\int_{\Omega} u_h(\mu) \cdot \Phi_i^h\ dx \textrm{ and } \alpha_i^H(\mu)=\int_{\Omega} u_H(\mu) \cdot \Phi_i^h\ dx.
\end{equation*}$$


The purpose is to create a rectification matrix that allows us to pass from the coarse coefficients to the fine ones. This implies that if the true solution is in the reduced space, then the NIRB method will give this true solution. 
We consider $Ntrain$ training snapshots in $\mathcal{G}$ (the ones we used to build our reduced basis). 
We define $\mathbf{A}\in \mathbb{R}^{Ntrain \times N}$ the matrix of the coarse coefficients and $\mathbf{B} \in \mathbb{R}^{Ntrain \times N}$ the one constructed from the fine coefficients such that 


$$\begin{equation*}
  \forall i=1,\cdots,N, \textrm{ and }  \forall \mu_k \in  \mathcal{G}, \quad  A_{k,i}=\alpha_i^H(\mu_k),\quad \textrm{and }  B_{k,i}=\alpha_i^h(\mu_k).
\end{equation*}$$

The approach is based on a regularized least-square method. Without regularization, its purpose is to minimize the error between the projection of the rectified approximation onto the basis and the optimal approximation as a function of the rectification matrix. Thus, let us introduce the rectification matrix $\mathbf{R}=(R_{i,j})_{1\leq i,j \leq N} \in \mathbb{R}^{N \times N}: X_h^N \to X_h^N$. 

The rectification step aims to find $\mathbf{R}$ minimizing


$$\begin{equation*}
\lVert \overset{N}{\underset{i=1}{\sum}} \alpha_i^h(\mu_k) \Phi_i^h -  \overset{N}{\underset{i=1}{\sum}}  \overset{N}{\underset{j=1}{\sum}} R_{i,j} \alpha_j^H(\mu_k) \Phi_i^h \rVert^2,\quad \forall k=1,\dots,Ntrain.
\end{equation*}$$


With the $L^2$ orthonormalization of the RB functions, it is equivalent to minimize


$$\begin{equation*}
\lvert  \alpha_i^h(\mu_k)-   \overset{N}{\underset{j=1}{\sum}} \ R_{i,j}\ \alpha_j^H(\mu_k)  \rvert^2, \quad \forall k=1,\dots,Ntrain.
\end{equation*}$$


as a function of $\mathbf{R}$.
Thus, it consists in looking for $\mathbf{R}$ minimizing the cost functions


$$\begin{equation*}
  \lVert \mathbf{A}\mathbf{R}_i-\mathbf{B}_i \rVert^2_{2},\quad  i=1,\cdots,N.
\end{equation*}$$

We use a Tikhonov regularization that consists in promoting solutions of such problems with small norms.
Thus it becomes


$$\begin{equation*}
  \lVert \mathbf{A}\mathbf{R}_i-\mathbf{B}_i \rVert^2_{2}+\lambda \lVert \mathbf{R}_i \rVert_2^2,\quad  i=1,\cdots,N,
\end{equation*}$$

where $\lambda$ is a regularization term.


The solution to this problem is the rectification matrix:


$$\begin{equation*}
              \mathbf{R}_i=(\mathbf{A}^T\mathbf{A}+\lambda \mathbf{I}_{N})^{-1}\mathbf{A}^T \mathbf{B}_i, \  i=1, \cdots,N,
   \end{equation*}$$      
  
Then, the NIRB approximation becomes

$$\begin{equation}
               Ru_{Hh}^N(\mu)=\overset{N}{\underset{i,j=1}{\sum}}\ R_{ij}\ \alpha_j^H(\mu)\ \Phi_i^h.
            \end{equation}$$




## References

- [Chakir, R., Maday, Y., Parnaudeau, P. (2019). A non-intrusive reduced basis approach for parametrized heat transfer problems. Journal of Computational Physics, 376, 617-633.](https://www.sciencedirect.com/science/article/pii/S0021999118306570?casa_token=ypCz1682c_QAAAAA:Lluz2uJZyqwPWNiRxEfPn-yVjAE1wO2-fLUjnQnYUq7OQ6rsvi4xfo9SxvGCd1WqjTjn3Ad_rFU)

- [Grosjean, E., & Maday, Y. (2021). Error estimate of the non-intrusive reduced basis method with finite volume schemes. ESAIM: Mathematical Modelling and Numerical Analysis, 55(5), 1941-1961](https://www.esaim-m2an.org/articles/m2an/abs/2021/06/m2an210043/m2an210043.html)