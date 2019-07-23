# Study on different means of flux computation

There is a related topic on the FEniCS discussion forum: See [here](https://fenicsproject.discourse.group/t/compute-gradient-of-scalar-field-on-boundarymesh/1172).

## Motivation

For Dirichlet-Neumann coupling we need to compute the flux at the coupling boundary from a given temperature field. For this task there exist different approaches with different accuracy. In this folder the approaches are evaluated for a simple heat equation -- which means that no partitioning or coupling is applied. Note that in the coupled simulation quality of the results may differ.

There are two main criteria for assessing the quality of the approach:

* Overall Conservation: compare the flux across the boundary with the sources inside the domain (Gauss theorem).

* Accuracy in the single nodes of the mesh: compare analytical solution for flux $q_x, q_y$ with numerical solution. Ideally, we want to reach machine precision.

## Setup and analytical solution

Our setup is derived from the [heat equation tutorial](https://fenicsproject.org/pub/tutorial/html/._ftut1006.html) in the FEniCS tutorials book.

We solve the heat equation $ u_t = \Delta u + f $. For a given right hand side $f=\beta - 2 - 2 \alpha$ we obtain the following analytical solution for temperature

$$ u = 1 + x^2 + \alpha y^2 + \beta t $$

and flux 

$$ q_x = 2 x; q_y = 2 \alpha y $$

In the following we study a number of solution procedures for this problem with special interest in the computation of the flux. All procedures (except mixedForm.py) use the same approach as explained in the tutorial. The flux is only computed as a post-processing step from the known solution for the temperature field.

## computingDerivatives.py

Procedure follows [this tutorial](http://hplgit.github.io/INF5620/doc/pub/fenics_tutorial1.1/tu2.html#tut-poisson-gradu). Overall flux is conserved and we can reach machine precision. Note that second order functions are used as basis functions for the temperature field (`V = FunctionSpace(mesh, 'P', 2)`), otherwise we do not reach machine precision.

## computingMoreDerivatives.py

Variant of *computingDerivatives.py*, where the same accuracy can be reached. Instead of a `VectorFunctionSpace` we use two scalar `FunctionSpace`s for flux reconstruction.

## domainWithCorners.py

Procedure follows [this suggestion](https://fenicsproject.discourse.group/t/compute-gradient-of-scalar-field-on-boundarymesh/1172/2). Overall flux is conserved. Accuracy in single nodes depends on mesh resolution.

## gradientOnBoundaryMesh.py

Procedure follows [this suggestion](https://fenicsproject.discourse.group/t/project-gradient-on-boundarymesh/262/2). Overall flux is conserved. Accuracy in single nodes depends on mesh resolution. However, this approach is less accurate than the above one (domainWithCorners.py)

## manually.py

Procedure loosely follows suggesiton on p.3 in Toselli, Andrea, and Olof Widlund. Domain decomposition methods-algorithms and theory. Vol. 34. Springer Science & Business Media, 2006. Has been tested succesfully in coupled scenarios, as well. However, the approach is mathematically not 100% clear and difficult to parallelize.

## mixedForm.py

The whole problem is solved using a mixed formulation. This means that the flux is not computed as a post-processing, but that computing the flux is actually part of the solution procedure. Overall flux is conserved and we can reach machine precision. 
