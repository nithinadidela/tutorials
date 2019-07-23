# based on https://fenicsproject.discourse.group/t/compute-gradient-of-scalar-field-on-boundarymesh/1172/2

from dolfin import *

Nel = 100
x_left, x_coupling = 0, 1
y_bottom, y_top = 0, 1
p0 = Point(x_left, y_bottom)
p1 = Point(x_coupling, y_top)
nx, ny = 20, 20
alpha = 3.0  # parameter alpha
beta = 1.3  # parameter beta
dt = 1

# Desired exact solution:
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
# Residual of the Poisson equation
f = Constant(beta - 2 - 2 * alpha)
f_N = Expression('2 * x[0]', degree=1)
mesh = RectangleMesh(p0, p1, nx, ny, diagonal='crossed')
n = FacetNormal(mesh)
V = FunctionSpace(mesh, "Lagrange", 1)
u_n = interpolate(u_D, V)
u_D.t = dt

# The full boundary, on which we apply a Dirichlet BC:
class Bdry(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


# Interior of the domain:
class Interior(SubDomain):
    def inside(self, x, on_boundary):
        return (not (near(x[0], 0) or near(x[0], 1)
                     or near(x[1], 0) or near(x[1], 1)))


# The boundary on the right side of the domain, on which we want to
# extract the flux:
class BdryOfInterest(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] > 1.0 - DOLFIN_EPS) and on_boundary


# Everything BUT the right side of the domain from which we want to
# extract the boundary flux; this must include all nodes not used to
# approximate flux on the boundary of interest, including those in the
# interior of the domain (so `on_boundary` should not be used in the
# return value).
class AntiBdry(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 1.0 - DOLFIN_EPS


# Mark parts of the boundary to which integrals will need to be restricted.
# (Parts not explicitly marked are flagged with zero.)
FLUX_BDRY = 1
COMPLEMENT_FLUX_BDRY = 2
boundaryMarkers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1,
                               COMPLEMENT_FLUX_BDRY)
BdryOfInterest().mark(boundaryMarkers, FLUX_BDRY)
ds_marked = ds(subdomain_data=boundaryMarkers)

# Boundary conditions to apply to solution, `u_h`:
BCs = [DirichletBC(V, u_D, Bdry()), ]

# Boundary conditions to apply when solving for flux on whole boundary:
interiorBCs = [DirichletBC(V, Constant(0.0), Interior()), ]

# Boundary conditions to apply to the flux solution when we are only
# interested in flux on the right side of the domain:
antiBCs = [DirichletBC(V, Constant(0.0), AntiBdry()), ]

def R(u, v):
    dudt = (u - u_n) / dt
    F = 0
    F += dot(grad(u), grad(v)) * dx
    F += dudt * v * dx
    F += -f * v * dx
    return F


u = TrialFunction(V)
v = TestFunction(V)
F = R(u, v)
a = lhs(F)
L = rhs(F)

# Solve and put solution in `u_h`
u_np1 = Function(V)
solve(a == L, u_np1, BCs)

# Solve in the trace space for what Neumann data, i.e.,
#
#  \mathbf{q}\cdot\mathbf{n} = \nabla u\cdot\mathbf{n}
#
# would have produced the solution from the Dirichlet problem; this is the
# notion of flux that satisfies the underlying conservation law.
qn = TrialFunction(V)

################################################################################

# The trick:  Since we want to use the corner nodes to approximate the
# flux on our boundary of interest, test functions will end up being
# nonzero on an $O(h)$ part of the complement of the boundary of interest.
# Thus we need to integrate a consistency term on that part of the boundary.

n = FacetNormal(mesh)
consistencyTerm = inner(grad(u_np1), n) * v * ds_marked(COMPLEMENT_FLUX_BDRY)
FBdry = qn * v * ds_marked(FLUX_BDRY) - R(u_np1, v) + consistencyTerm

FBdry_inconsistent = qn * v * ds_marked(FLUX_BDRY) - R(u_np1, v)

# Applying the consistent flux extraction on the full boundary
# and then restricting the result is also sub-optimal; this flux extraction
# technique doesn't appear to play nicely with corners.
FBdry_full = qn * v * ds - R(u_np1, v)


################################################################################

# Get $\mathbf{q}\cdot\mathbf{n}$ on the boundary of interest with and
# without the consistency term:
def solveFor_qn_h(FBdry, BCs):
    aBdry = lhs(FBdry)
    LBdry = rhs(FBdry)
    ABdry = assemble(aBdry, keep_diagonal=True)
    bBdry = assemble(LBdry)
    [BC.apply(ABdry, bBdry) for BC in BCs]
    qn_h = Function(V)
    solve(ABdry, qn_h.vector(), bBdry)
    return qn_h


qn_h = solveFor_qn_h(FBdry, antiBCs)
qn_h_inconsistent = solveFor_qn_h(FBdry_inconsistent, antiBCs)
qn_h_full = solveFor_qn_h(FBdry_full, interiorBCs)

# Compare fluxes with the exact solution:
import math


def fluxErr(qn):
    err = inner(grad(u_D), n) - qn
    return math.sqrt(assemble(err * err * ds_marked(FLUX_BDRY)))


# Converges at 3/2 order in $L^2$, as expected for smooth solutions, according
# to this paper:
#
#  https://link.springer.com/article/10.1007/BF01385871
#
"""
print("Error in consistent flux on boundary of interest: "
      + str(fluxErr(qn_h)))

# Converges at only 1/2 order:
print("Error in restricted consistent flux from full boundary: "
      + str(fluxErr(qn_h_full)))

# Converges at only 1/2 order:
print("Error in inconsistent flux on boundary of interst: "
      + str(fluxErr(qn_h_inconsistent)))

# Converges at first order:
print("Error in direct flux: " + str(fluxErr(inner(grad(u_np1), n))))
"""
# Slow for large meshes; also first-order:
# print("Error in projection to linears: "
#      +str(fluxErr(inner(project(grad(u_h),
#                                 VectorFunctionSpace(mesh,"CG",1)),n))))

import numpy as np
from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(np.linspace(0, 1), [qn_h_full(x_left, y) for y in np.linspace(0, 1)], ':', label="full left edge")
plt.plot(np.linspace(1, 2), [qn_h_full(x, y_top) for x in np.linspace(0, 1)], ':', label="full top edge")
plt.plot(np.linspace(2, 3), [qn_h_full(x_coupling, y) for y in np.linspace(1, 0)], ':', label="full right edge")
plt.plot(np.linspace(3, 4), [qn_h_full(x, y_bottom) for x in np.linspace(1, 0)], ':', label="full bottom edge")
#plt.plot(np.linspace(0, 1), [qn_h(x_left, y) for y in np.linspace(0, 1)], label="left edge")
#plt.plot(np.linspace(1, 2), [qn_h(x, y_top) for x in np.linspace(0, 1)], label="top edge")
plt.plot(np.linspace(2, 3), [qn_h(x_coupling, y) for y in np.linspace(1, 0)], '-x', label="consistent right edge")
plt.plot(np.linspace(2, 3), [qn_h_inconsistent(x_coupling, y) for y in np.linspace(1, 0)], '-o', label="inconsistent right edge")
#plt.plot(np.linspace(3, 4), [qn_h(x, y_bottom) for x in np.linspace(1, 0)], label="bottom edge")
plt.legend()
plt.ylabel('normal heat flux')
plt.xlabel('arc length')

print([qn_h(x_coupling, y) for y in np.linspace(1, 0)])

#plot(mesh)
#plot(BoundaryMesh(mesh, "exterior"))
plt.figure(2)
plot(u_np1)
x, y = map(list, zip(*[(x_left - .1 * qn_h(x_left, y), y) for y in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_top + .1 * qn_h(x, y_top)) for x in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x_coupling + .1 * qn_h(x_coupling, y), y) for y in np.linspace(1, 0)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_bottom - .1 * qn_h(x, y_bottom)) for x in np.linspace(1, 0)]))
plt.plot(x, y)
plt.show()