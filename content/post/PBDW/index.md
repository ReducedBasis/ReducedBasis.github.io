---
# Documentation: https://docs.hugoblox.com/managing-content/

title: ""
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2024-12-13T10:19:20+01:00
lastmod: 2024-12-13T10:19:20+01:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
# THE PBDW method



```python
import sys
!{sys.executable} -m pip install numpy
!{sys.executable} -m pip install matplotlib
!{sys.executable} -m pip install scikit-fem
```

#### Let us present one method that combines model order reduction and a data assimilation problem: the Parametrized-Background Data-Weak method (PBDW). 
The PBDW method exploits data observations and the knowledge of a parameterized best-knowledge (bk) model that describes the physical system, to improve performance. We denote by $u^{bk}(\mu) \in \mathcal{U}$, the solution to the parameterized model for the parameter value $\mu \in \mathcal{P}^{bk}$, 
                                                                                                                                             $$ G^{bk,\mu}(u^{bk}(\mu)) = 0.$$
Here, $G^{bk,\mu}(\cdot)$ denotes the parameterized bk model associated with the system, and $\mathcal{P}^{bk} \subset \mathbb{R}^P$ is a compact set that reflects the lack of knowledge in the value of the model parameters. We further define the bk solution manifold
$$ \mathcal{M}^{bk} = \{u^{bk}(\mu) : \mu \in \mathcal{P}^{bk}\}.$$

The PBDW formulation integrates the parameterized mathematical model $G^{pb}$ and $M$ experimental observations associated with a parameter configuration $\mu^{true}$ to estimate the true field $u^{true}(\mu^{true})$ as well as any desired output $l_{out}(u^{true}(\mu^{true}\
)) \in \mathbb{R}$ for given output functional $l_{out}$.                                                                                                                  
We intend that $\lVert u^{true}(\mu^{true})-u^{bk}(\mu^{true}) \rVert$ is small (i.e. that our model represents our data observations well).

The PBDW is decomposed in two parts: one offline and one online.

In the following notebook, we will employ:
- as our model problem a 2D advection-diffusion problem,
- some measures that could also be possibly be noisy (here we will consider the case with no noise),
- a sequence of background spaces that reflect our (prior) best knowledge model bk, 
- gaussians for our observations with a proper choice of localization.
  
We will create measures artificially from our model problem by modifying the right hand side of our equation and the boundary conditions.

To generate the sequence of background spaces, we will employ a POD basis (but any other offline algorithm could be used such as a weak-Greedy procedure). We will denote these spaces by 
$$\mathcal{Z}_1 \subset \dots \subset \mathcal{Z}_{N} \subset \mathcal{U}. $$                                                                                                                                                                                                              
To sum up, we consider two main ingredients:                                                                                                                                                                                                               
- a model with a bias leading to the $(\mathcal{Z}_n)_n$ sequence,                                                                                                                                                                        
- and true measures, that might be noisy.                                                                                                                                                                                               
                                                                                                                                              


```python
# import packages
import skfem  # for Finite Element Method
import numpy as np
import matplotlib.pyplot as plt
```

## The 2D advection-diffusion model problem:

We are going to use in this example a 2D  advection-diffusion problem with the Finite Element Method (FEM), which consists in 
solving on a unit square (denoted $\Omega$) the following equations:
$$\begin{align*}
&- \Delta u + b(\mu) \cdot \nabla u =x_1 x_2 + g_1, \textrm{ in } \Omega,\\
& u=4 x_2(1-x_2)(1+g_2), \textrm{ on } \Omega_{left}:=\partial \Omega \cap \{x=0\},\\
& \partial_n u (\mu)=0, \textrm{ on } \partial \Omega \backslash \Omega_{left},
\end{align*}$$
where $u \in H^1(\Omega)$ represents the unknown, $\mu \in \mathbb{R}^2$ is our variable parameter, and $g_1$ and $g_2$ are two functions that allow us to construct a model with a bias with respect to the data observations. The lack of knowledge of the value of $\mu$ constitutes the anticipated parametric ignorance in the model, while uncertainty in $g_1$ and $g_2$ constitutes the unanticipated non-parametric ignorance.

We employ $P1$ finite elements to get a proper solution and obtain the system $\mathbf{A} \mathbf{x} =\mathbf{f}$ to solve where the assembled matrix $\mathbf{A}$ corresponds to the bilinear part $\int_{\Omega} (\nabla u, \nabla v) - (b(\mu),\nabla v) u + \int_{\Omega_{left}} (b(\mu), \nabla_n u) v $.

The dirichlet boundary conditions are imposed with a penalization method, called by the line: 
solve(*condense(K,f, x=uvp, D=D)),

where x=uvp gives the values at the boundaries of the velocity and D refers to the boundary decomposition.

This problem model comes from the paper "PBDW method for state estimation: error analysis for noisy data and non-linear formulation"(https://arxiv.org/abs/1906.00810).




```python
# First we define a mesh for the unit square
mesh= skfem.MeshTri().refined(5).with_boundaries(                                                                
        {                                                                                                                                
            "left": lambda x: x[0] == 0,                                                                                            
            "right": lambda x: x[0] == 1,            
            "down": lambda x: x[1] == 0,                                                                                            
            "up": lambda x: x[1] == 1,     
            
        }                                                                                                                               
)

print(mesh)
mesh

```

    <skfem MeshTri1 object>
      Number of elements: 2048
      Number of vertices: 1089
      Number of nodes: 1089
      Named boundaries [# facets]: left [32], bottom [32], right [32], top [32], down [32], up [32]





    
![svg](./PBDW_5_1.svg)
    




```python
"""First step."""

# Assembling matrices

from skfem.assembly import BilinearForm, LinearForm
from skfem.helpers import grad, dot

# Bilinear form 
@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))

@BilinearForm
def gradu(u, v, w):    
    return - u*(v.grad[0]*w.bmu1+v.grad[1]*w.bmu2) 

@BilinearForm
def boundary_normal_gradient(u, v, w):
    """ Bilinear part for int(grad_n(u), v) on the left boundary """
    return dot(grad(u), w.n) * v

@BilinearForm
def mass(u, v, _): #H1 norm (norm for the space U)
    return u * v +dot(grad(u), grad(v))

@BilinearForm
def L2mass(u, v, _): #L2 norm
    return u * v 
    
  
```

In the code bellow, the function SolveAdvDiffProblem([mu1,mu2],Mesh,measures) takes as inputs the parameter $\mu \in \mathcal{P}^{bk} \subset \mathbb{R}^2$, a mesh, and a boolean used for the bias on the right-hand side function and on the dirichlet boundary conditions, and it returns the associated solution.


```python
from skfem import *
# Compute solution 
element = ElementTriP1() #P1 FEM elements

def SolveAdvDiffProblem(mu,Mesh,measures):
    # mu: variable parameter
    # mesh: FE mesh
    # measures: Boolean (if true, the right-hand side function and dirichlet parameter are adapted to the generation of the data observations)
    basis = Basis(Mesh, element) 
    #print('Parameter mu:',mu)
    mu1,mu2=mu[0],mu[1]
    bmu1= lambda x: mu[0]*(np.cos(mu[1])) 
    bmu2= lambda x: mu[0]*(np.sin(mu[1])) 
    bmu1=basis.project(bmu1)
    bmu2=basis.project(bmu2)
    
    ## Assembling global problem
    A= laplace.assemble(basis)
    A2=gradu.assemble(basis,bmu1=basis.interpolate(bmu1),bmu2=basis.interpolate(bmu2))
    A+=A2

    ## boundary conditions
    left_basis = FacetBasis(Mesh, element, facets=Mesh.boundaries['right'])
    
    def profil_left(x):
        # Dirichlet boundary conditions
        if measures==True:
            #return (x[0]==1)*200*x[1]#(8*x[1]*(1-x[1])*(1+ np.sin(2*np.pi*x[1]))) #with g2!=0
            return (x[0]==1)*(4*x[1]*(1-x[1])*(1+ np.sin(2*np.pi*x[1]))) #with g2!=0
        else:
            #return (x[0]==1)*100*x[1]#(4*x[1]*(1-x[1]))  #with g2=0
            return (x[0]==1)*(4*x[1]*(1-x[1]))  #with g2=0

    u_boundary=left_basis.project(profil_left)
    D = basis.get_dofs(['right'])
    ## Neumann boundary conditions 
    A_boundary = boundary_normal_gradient.assemble(left_basis)
    A+=A_boundary

    ## Right hand side functions
    if measures==True:
        #g1 = lambda x: x[0]**2 -x[1] #with g1!=0
        g1 = lambda x: 0.2*x[0]**2 +x[0]*x[1] #with g1!=0
        g_proj1 = basis.project(g1)
    else:
        #g1 = lambda x: -x[1] #with g1=0
        g1 = lambda x: x[0]*x[1] #with g1!=0
        g_proj1 = basis.project(g1)
        
    f=g_proj1
    u = solve(*condense(A,f, x=u_boundary, D=D))

    return u
    

```


```python
## plot the solution

from skfem.visuals.matplotlib import plot, draw, savefig

u=SolveAdvDiffProblem([0.4,np.pi/6],mesh,True) #
plot(mesh, u, shading='gouraud', colorbar=True)

u=SolveAdvDiffProblem([0.4,np.pi/6],mesh,False) # examples
plot(mesh, u, shading='gouraud', colorbar=True)

```




    <Axes: >




    
![png](./PBDW_9_1.png)
    



    
![png](./PBDW_9_2.png)
    


## The PBDW method

We are now able to proceed with the offline and online parts of the PBDW method.
We start with a classical POD algorithm on the bk model with bias, but first we construct the data observations $u^{true}(\mu^{true})$ artificially (here we consider that we have access to the data in the whole domain).

### OFFLINE PART


We define one fine mesh called "FineMesh" in the code


```python
## FINE MESH
FineMesh = skfem.MeshTri().refined(5).with_boundaries(                                                                
        {                                                                                                                                
            "left": lambda x: x[0] == 0,                                                                                            
            "right": lambda x: x[0] == 1,            
            "down": lambda x: x[1] == 0,                                                                                            
            "up": lambda x: x[1] == 1,     
            
        }                                                                                                                               
)
FineBasis = Basis(FineMesh, element)

NumberOfNodesFineMesh = FineMesh.p.shape[1]
print("number of nodes: ",NumberOfNodesFineMesh)
num_dofs_uFineMesh = FineBasis.doflocs.shape[1] # or np.shape(u)[0] for DOFs
print("number of DoFs: ",num_dofs_uFineMesh)

massMat=mass.assemble(FineBasis)
L2massMat=L2mass.assemble(FineBasis)
```

    number of nodes:  1089
    number of DoFs:  1089



```python
utrue=SolveAdvDiffProblem([1.4,np.pi/3],mesh,True) # True for measures
plot(FineMesh, utrue, shading='gouraud', colorbar=True)
```




    <Axes: >




    
![png](./PBDW_14_1.png)
    



```python

```

#### POD on bk model


```python
""" POD on the bk model (could be done with a greedy approach) """
print("-----------------------------------")
print("        Offline POD                ")
print("-----------------------------------")

NumberOfSnapshots=0
NumberOfModesN=15
print("number of modes for the bk model: ",NumberOfModesN)

mu1list=np.linspace(0.1,3,10)
mu2list=np.linspace(0,np.pi/2,10) 

FineSnapshots=[]

for mu1 in (mu1list):
    for mu2 in (mu2list):
        u=SolveAdvDiffProblem([mu1,mu2],FineMesh,False)
        FineSnapshots.append(u)
        NumberOfSnapshots+=1

## SVD ##
 
# We first compute the correlation matrix C_ij = (u_i,u_j)_U
CorrelationMatrix = np.zeros((NumberOfSnapshots, NumberOfSnapshots))
for i, snapshot1 in enumerate(FineSnapshots):
    MatVecProduct = massMat.dot(snapshot1)
    for j, snapshot2 in enumerate(FineSnapshots):
        if i >= j:
            CorrelationMatrix[i, j] = np.dot(MatVecProduct, snapshot2)
            CorrelationMatrix[j, i] = CorrelationMatrix[i, j]

# Then, we compute the eigenvalues/eigenvectors of C 
EigenValues, EigenVectors = np.linalg.eigh(CorrelationMatrix, UPLO="L") #SVD: C eigenVectors=eigenValues eigenVectors
idx = EigenValues.argsort()[::-1] # sort the eigenvalues

TotEigenValues = EigenValues[idx]
TotEigenVectors = EigenVectors[:, idx]
EigenValues=TotEigenValues[0:NumberOfModesN]

EigenVectors=TotEigenVectors[:,0:NumberOfModesN]
print("eigenvalues: ",EigenValues)

RIC=1-np.sum(EigenValues)/np.sum(TotEigenValues) #must be close to 0
print("Relativ Information Content:",RIC)

ChangeOfBasisMatrix = np.zeros((NumberOfModesN,NumberOfSnapshots))

for j in range(NumberOfModesN):#orthonormalization
    ChangeOfBasisMatrix[j,:] = EigenVectors[:,j]/np.sqrt(EigenValues[j])

bkReducedBasis = np.dot(ChangeOfBasisMatrix,FineSnapshots)

```

    -----------------------------------
            Offline POD                
    -----------------------------------
    number of modes for the bk model:  15
    eigenvalues:  [2.03866121e+06 1.98788941e+05 7.63460590e+04 3.57206711e+03
     3.26553742e+03 1.08125000e+03 4.08015648e+01 3.47784483e+01
     2.37761217e+01 4.40644071e+00 3.45055983e-01 1.56321411e-01
     1.29748847e-01 6.77589026e-02 9.30473722e-03]
    Relativ Information Content: 1.3066825399477011e-09


Now we can check for the reduced basis accuracy.


```python
print("-----------------------------------")
print("             Reduced basis accuracy")
print("-----------------------------------")
### Offline Errors
print("Offline Errors")
CompressionErrors=[]
H1compressionErrors=[]

#for snap in FineSnapshots:
#    ExactSolution =snap
for mu1 in (mu1list):
    for mu2 in (mu2list):
        ExactSolution=SolveAdvDiffProblem([mu1,mu2],FineMesh,False)
        #ExactSolutionbis=SolveAdvDiffProblem([mu1,mu2],FineMesh,False)
        CompressedSolutionU= ExactSolution@(massMat@bkReducedBasis.transpose())
        ReconstructedCompressedSolution = np.dot(CompressedSolutionU, bkReducedBasis) #pas de tps 0
    
        norml2ExactSolution=np.sqrt(ExactSolution@(L2massMat@ExactSolution))
        normh1ExactSolution=np.sqrt(ExactSolution@(massMat@ExactSolution))
        t=ReconstructedCompressedSolution-ExactSolution
    
        if norml2ExactSolution !=0 and normh1ExactSolution != 0:
            relError=np.sqrt(t@L2massMat@t)/norml2ExactSolution
            relh1Error=np.sqrt(t@massMat@t)/normh1ExactSolution
        else:
            relError = np.linalg.norm(ReconstructedCompressedSolution-ExactSolution)
        CompressionErrors.append(relError)
        H1compressionErrors.append(relh1Error)
        if mu1==2.275 and mu2==0:
            plot(FineMesh, ExactSolution, shading='gouraud', colorbar=True)
            plot(FineMesh, ExactSolutionbis, shading='gouraud', colorbar=True)
            plot(FineMesh, ReconstructedCompressedSolution, shading='gouraud', colorbar=True)
print("L2 compression error =", CompressionErrors)
print("H1 compression error =", H1compressionErrors)



```

    -----------------------------------
                 Reduced basis accuracy
    -----------------------------------
    Offline Errors
    L2 compression error = [4.464044518872249e-06, 3.2948510494119077e-06, 2.290250760263449e-06, 1.4675605306478524e-06, 8.915840966017943e-07, 7.557337160345698e-07, 1.0344107189534399e-06, 1.4292838150362181e-06, 1.839008459568471e-06, 2.260403555595182e-06, 5.6419137062022195e-06, 3.0292521469093515e-06, 2.638316514323157e-06, 2.9735064128926414e-06, 3.1133058130523887e-06, 3.071241432378604e-06, 3.0903399820237977e-06, 3.4597258065352133e-06, 4.385570343265425e-06, 5.914766889195202e-06, 6.466039662035052e-06, 2.2828281714074417e-06, 3.719265675449631e-06, 4.07665068527531e-06, 3.733074907693582e-06, 3.3228223211801526e-06, 3.0466976159325082e-06, 2.8139328856957787e-06, 2.7628176702476534e-06, 4.417740843515894e-06, 6.878018194786359e-06, 4.343622954486847e-06, 5.6479455139503074e-06, 4.983955485533267e-06, 4.643064369149396e-06, 4.356187524270617e-06, 4.068903388344752e-06, 4.325844963914229e-06, 4.463838149770944e-06, 5.069891550394366e-06, 7.252968845644662e-06, 7.1161760893413015e-06, 5.825223666363488e-06, 4.9078655698982395e-06, 6.477105587901556e-06, 5.629623181915566e-06, 3.435985246077469e-06, 3.666591301006391e-06, 4.3126374326462805e-06, 4.930810679681774e-06, 7.765028936646113e-06, 1.0758022999484722e-05, 5.765204954875644e-06, 6.881897092663532e-06, 9.823215026672102e-06, 7.545388982775257e-06, 4.999511761920188e-06, 6.2501806939502245e-06, 5.956618838091266e-06, 5.393592227066799e-06, 8.420860575531317e-06, 1.530055486222261e-05, 6.4263504016545675e-06, 8.70585220712872e-06, 1.093564578021148e-05, 5.531485466266688e-06, 4.88248204190274e-06, 7.536660217951777e-06, 5.605723600322189e-06, 6.0522794976476525e-06, 1.0317064718671453e-05, 2.021801920620293e-05, 7.932386218640109e-06, 1.0181978031918473e-05, 1.1198354134825336e-05, 5.04031183625094e-06, 8.884774594090558e-06, 8.337725622089321e-06, 3.784191171541323e-06, 7.740234225023708e-06, 1.7396051662626773e-05, 2.490089983737211e-05, 1.0177261735907454e-05, 1.2420205045018331e-05, 1.2400700091772858e-05, 9.107295330219943e-06, 1.322879798530576e-05, 8.462896334417241e-06, 8.136017146956131e-06, 7.40573172752598e-06, 3.319480686662823e-05, 2.8865070992548302e-05, 1.2764357020348877e-05, 1.617969257609354e-05, 1.3977194466545846e-05, 1.1508196930328714e-05, 1.4325893468926336e-05, 9.600712833369026e-06, 1.5911312993681602e-05, 1.1185720850111092e-05]
    H1 compression error = [2.7252932301889785e-05, 2.013750683906652e-05, 1.402739794153043e-05, 9.03203033440891e-06, 5.587308491638867e-06, 4.914693737670023e-06, 6.770478212303681e-06, 9.377848140674137e-06, 1.2148886270479421e-05, 1.5068140469183008e-05, 3.538305427019371e-05, 1.9688743754521856e-05, 1.6012708785290326e-05, 1.733984297032835e-05, 1.82353361047264e-05, 1.8227299129936916e-05, 1.8412505197389516e-05, 2.0446903618376864e-05, 2.592173938409032e-05, 3.554656059492433e-05, 3.796273452215796e-05, 1.4436918887135222e-05, 2.092649864639411e-05, 2.3683363675915002e-05, 2.2848548422432317e-05, 2.1207231093055237e-05, 1.9563234268424123e-05, 1.7360600445174253e-05, 1.6053761582722742e-05, 2.7579187022371675e-05, 3.8786741840616874e-05, 2.2619722066470402e-05, 3.0113021819023792e-05, 2.82996368838738e-05, 2.656205066435831e-05, 2.4652490908009468e-05, 2.3350583231630944e-05, 2.454448822024161e-05, 2.4121676619653384e-05, 3.0152660068145523e-05, 4.10866622034923e-05, 3.35404485790202e-05, 3.063462381567499e-05, 2.689789093972627e-05, 3.151552824445099e-05, 2.6637387395390685e-05, 1.79925538977804e-05, 2.1420579255796295e-05, 2.294970681598286e-05, 2.71096597917859e-05, 4.436459028061381e-05, 4.8435418288719186e-05, 3.0530069439965646e-05, 3.176041176607239e-05, 4.273556464930769e-05, 3.2664926942419514e-05, 2.3358513419411886e-05, 3.154113689298677e-05, 2.9969324371075966e-05, 2.759118381889585e-05, 4.675240980636919e-05, 6.692728464979316e-05, 3.213983174315306e-05, 3.521510054042893e-05, 4.534984712394246e-05, 2.3582656573518707e-05, 2.1276223191861312e-05, 3.4513081768172264e-05, 2.7845065891343934e-05, 2.9574001635377192e-05, 5.137634592785078e-05, 8.61508367731696e-05, 3.6750504542604e-05, 4.031827349791773e-05, 4.714785992850797e-05, 2.1638631599125338e-05, 3.3666719164020035e-05, 3.3455551337854464e-05, 1.983379959723556e-05, 3.3242891489082104e-05, 7.50107840515417e-05, 0.0001031623106532193, 4.529839464211873e-05, 5.104232741789849e-05, 5.126545848318776e-05, 3.3064402356663036e-05, 4.785757709236215e-05, 3.1064457959244826e-05, 3.314365823126774e-05, 2.8298730472408282e-05, 0.00013533770789244366, 0.0001159802382139368, 5.6097565167957744e-05, 6.732256071877919e-05, 5.852998165365517e-05, 5.1157193654252605e-05, 5.843907884382997e-05, 3.895991570740153e-05, 6.118099709629903e-05, 4.0914896868423286e-05]


#### Construction of the sensors                                                                       
We now characterize our measures:

We will consider $M$ data  $y^{obs}(\mu^{true})\in \mathbb{C}^M$, obtained from sensors at different localizations such that:                                                                              
 $$\forall m=1,\dots,M, y_m^{obs}=l^0_m(u^{true}(\mu^{true})).$$                                                    
Here $y_m^{obs}(\mu^{true})$ is the value of the $m$-th observation, with $l_m^0 \in \mathcal{U}'$.                                          The form of the observation functional depends on the specific transducer used to acquire data. For instance, if the transducer measures a local state value, then we may model the observation as a Gaussian convolution (like in this example).

We first associate with each observation functional $l_m^0 \in \mathcal{U}'$ an observation function                                                                                                                                            
$$\forall m=1,\dots,M, \ q_m = R_u l^0_m,$$                                                                                                                                                                                            
which is the Riesz representation of the functional. 
The condition $(q_m,v)=l_m^0(v)$ is equivalent to the resolution of the following problem: 

Find $q_m \in \mathcal{U}$ such that:
$M q_m=b$ with $M $ the mass matrix and $b$ the vector obtained from the observation functionals $b_i=l_m^0(v_i)$ where $v_i$ are the FEM test functions. 

Then we introduce hierarchical observation spaces $\mathcal{U}_M \subset \mathcal{U}$ such that  
$$\forall M=1,\dots,M_{max},\ \mathcal{U}_M=\textrm{Span} \{ q_m\}_{m=1}^M. $$

We want to choose the localization of the sensors adequately in order to get $M$ small such that $(q_m)_{m=1}^M \in (\mathcal{U}_M)^M$ forms a canonical reduced basis of the observation spaces.

In our example, we consider Gaussian observation functionals with standard deviation $r_w = 0.01$:
$$l_m(v)=l_m(v,x_m,r_w)=C \int_{\Omega} exp \Big(-\frac{1}{2r_w^2}\lVert x-x_m \rVert_2^2 \Big) v(x) \ \textrm{dx},$$
where $C$ is such that $l_m(1)=1$, and $x_m$ corresponds to the center of the $m$-sensor localization.
In this example, we take these centers as the nodes of a coarser mesh but we could have used a GEIM to choose them more wisely.






```python
rw = 0.01  # deviation r_w
xm = np.array([0.5, 0.5]) #for test

## Normalization of the gaussian functionals
## lm = C exp(-||x - x_m||^2 / (2 * r_w^2))
def gaussian_weight_noC(x,xm):
    """ Gaussian function centered on xm of deviation rw (to find C for normalization). """
    return np.exp(-0.5 * np.sum((x - xm[:, np.newaxis,np.newaxis])**2, axis=0) / rw**2)

def compute_C(xm):
    # takes xm center of sensor as parameter
    
    @LinearForm
    def lm(v, w):
        """ Computes lm(1) by integrating the Gaussian weight. """
 
        weight = gaussian_weight_noC(w.x,xm)
        return weight * v 

    # Assemble the linear form for lm(1)
    b = lm.assemble(FineBasis)
    lm_value = np.sum(b)  # Numerical integration result: b@1 since v=1
    
    C = 1.0 / lm_value  # Scaling factor
    return C

C = compute_C(xm)

```


```python
# lm = C exp(-||x - x_m||^2 / (2 * r_w^2))
def gaussian_weight(x,xm,C):
    """ Gaussian function centered on xm of deviation rw """
    return C*np.exp(-0.5 * np.sum((x - xm[:, np.newaxis,np.newaxis])**2, axis=0) / rw**2)

def qm_b(xm,C):
    # weak formulation
    # mass matrix : (v, q_m)_U
    
    @LinearForm
    def rhs(v, w):
        """ Right hand side  """
        weight = gaussian_weight(w.x,xm,C)  # Applique la gaussienne
        return weight * v

    # Assembling
    b = rhs.assemble(FineBasis)

    # Solving the linear system
    from scipy.sparse.linalg import spsolve
    #from scipy.sparse.linalg import cg
    #q_m, info = cg(M, b)
    #if info == 0:
    #    print("Conjugate Gradient converged!")
    #else:
    #    print("Conjugate Gradient did not converge.")
    
    q_m = spsolve(massMat, b)  # Resolution of M @ q_m = b
    return q_m
```


```python
## sensors localizations from a coarser mesh M=25
## Coarse MESH
CoarseMesh = skfem.MeshTri().refined(3).with_boundaries(                                                                
        {                                                                                                                                
            "left": lambda x: x[0] == 0,                                                                                            
            "right": lambda x: x[0] == 1,            
            "down": lambda x: x[1] == 0,                                                                                            
            "up": lambda x: x[1] == 1,     
            
        }                                                                                                                               
)

xm_grid=CoarseMesh.p #xm = nodes

qmReducedBasis=np.zeros((num_dofs_uFineMesh,CoarseMesh.p.shape[1]))
for i in range(CoarseMesh.p.shape[1]):
    xm=xm_grid[:,i] #update xm
    C = compute_C(xm) # find C for normalization
    qmReducedBasis[:,i]=qm_b(xm,C) # generate qm
        
plot(FineMesh, qmReducedBasis[:,0], shading='gouraud', colorbar=True)
 
    
NumberOfModesM=np.shape(qmReducedBasis)[1]
```


    
![png](./PBDW_23_0.png)
    


Now we have two kind of basis functions:
- the $\Phi_i \in \mathcal{Z}_N$ for $i=1,\dots,N$ (in the code bkReducedBasis),
- the $q_j \in \mathcal{U}_M$, for $j=1,\dots,M$ (in the code qmReducedBasis).

#### PBDW algebraic form
We now state the algebraic form of the PBDW problem.
We need to compute  $(A)_{i,j}=(q_i,q_j)$ and $(B)_{i,j}=(\Phi_j,q_i)$ and to assemble $K= \begin{pmatrix} A & B \\
B^T & 0 \end{pmatrix}$ in order to solve $\begin{pmatrix} \eta_M \\ z_N \end{pmatrix} = K^{-1} \begin{pmatrix} y^{obs} \\ 0 \end{pmatrix}$


```python
print("---------------------------------------")
print("---- OFFLINE PART: BUILDING MATRICES --")
print("---------------------------------------")

############################################
## Assembling A and B  #####################
############################################

AMat=massMat
BMat=massMat

#############################################
## Project matrices onto the reduced space ##
#############################################

Areduced = np.zeros((NumberOfModesM, NumberOfModesM))
Breduced = np.zeros((NumberOfModesM, NumberOfModesN))

for i in range(NumberOfModesM):
    MatAVecProduct = AMat.dot(qmReducedBasis[:,i])
    for j in range(NumberOfModesM):
        if j>=i:
            Areduced[i,j] = np.dot(qmReducedBasis[:,j],MatAVecProduct)
            Areduced[j,i] = Areduced[i,j]

for i in range(NumberOfModesM):
    MatBVecProduct = BMat.dot(qmReducedBasis[:,i])
    for j in range(NumberOfModesN):
        Breduced[i,j] = np.dot(bkReducedBasis[j],MatBVecProduct)
         
            

Kreduced = np.zeros((NumberOfModesN+NumberOfModesM,NumberOfModesN+ NumberOfModesM))
#fill global K
for i in range(NumberOfModesM):
    for j in range(NumberOfModesM):
        Kreduced[i,j]=Areduced[i,j]

for i in range(NumberOfModesM):
    for j in range(NumberOfModesN):
        Kreduced[i,NumberOfModesM+j]=Breduced[i,j]
        Kreduced[NumberOfModesM+j,i]=Breduced[i,j]


```

    ---------------------------------------
    ---- OFFLINE PART: BUILDING MATRICES --
    ---------------------------------------


#### ONLINE PART

We can now solve $\begin{pmatrix} \eta_M \\ z_N \end{pmatrix} = K^{-1} \begin{pmatrix} y^{obs} \\ 0 \end{pmatrix}$
and we approximate $u^{true}(\mu^{true})$ by $u=\sum_{i=1}^M (\eta_M)_i \ q_i +\sum_{i=1}^N (z_N)_i \ \Phi_i $.


```python
# Construct the observations from the basis (qm)_m and the true data
yobs = np.zeros(NumberOfModesM)

#mu1=mu1list[1]
#mu2=mu2list[2]
#utrue=SolveAdvDiffProblem([mu1,mu2],FineMesh,True)
# Compute (q_m, v)_U
qm_inner =utrue@ massMat

for i in range(NumberOfModesM):    
    yobs[i] =qm_inner.dot(qmReducedBasis[:,i])


print("--------------------------------------------")
print("---- ONLINE PART: SOLVING REDUCED PROBLEM --")
print("--------------------------------------------")

RightHandSide= np.concatenate([yobs, np.zeros(NumberOfModesN)]) 
## SOLVING PROBLEM ##
#EtaZ=np.linalg.solve(Kreduced,RightHandSide)
#Eta, Z = EtaZ[:NumberOfModesM],EtaZ[NumberOfModesM:]
#print("norm",np.linalg.norm(Areduced@Eta+Breduced@Z-yobs))
#print("norm",np.linalg.norm(Breduced.transpose()@Eta))
#u=np.dot(Eta,qmReducedBasis.transpose())+np.dot(Z,bkReducedBasis)

Z=np.linalg.solve(np.dot(np.dot(Breduced.transpose(),np.linalg.inv(Areduced)),Breduced),np.dot(np.dot(Breduced.transpose(),np.linalg.inv(Areduced)),yobs))
Eta=np.linalg.solve(Areduced,yobs-np.dot(Breduced,Z))
u=np.dot(Eta,qmReducedBasis.transpose())+np.dot(Z,bkReducedBasis)

ExactSolution =utrue #exact observations
ReconstructedCompressedSolution = u #reduced solution
    
norml2ExactSolution=np.sqrt(ExactSolution@(L2massMat@ExactSolution))
normh1ExactSolution=np.sqrt(ExactSolution@(massMat@ExactSolution))
t=np.abs(ReconstructedCompressedSolution-ExactSolution)

if norml2ExactSolution !=0 and normh1ExactSolution != 0:
    relError=np.sqrt(t@L2massMat@t)/norml2ExactSolution
    relh1Error=np.sqrt(t@massMat@t)/normh1ExactSolution
    
else:
    relError = np.linalg.norm(ReconstructedCompressedSolution-ExactSolution)
    
print("L2 relative error: ",relError)
print("H1 relative error: ",relh1Error)

plot(FineMesh,utrue, shading='gouraud', colorbar=True)
plot(FineMesh,u, shading='gouraud', colorbar=True)
#plot(FineMesh,np.dot(Z,bkReducedBasis)-utrue, shading='gouraud', colorbar=True)
plot(FineMesh,u-utrue, shading='gouraud', colorbar=True)
```

    --------------------------------------------
    ---- ONLINE PART: SOLVING REDUCED PROBLEM --
    --------------------------------------------
    L2 relative error:  0.0004129386415760589
    H1 relative error:  0.003935527762515732





    <Axes: >




    
![png](./PBDW_27_2.png)
    



    
![png](./PBDW_27_3.png)
    



    
![png](./PBDW_27_4.png)
    



```python

```
