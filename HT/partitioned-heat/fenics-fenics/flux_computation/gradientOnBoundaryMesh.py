# extremely helpful: https://fenicsproject.discourse.group/t/project-gradient-on-boundarymesh/262

from fenics import Function, RectangleMesh, FunctionSpace, Point, Expression, Constant, DirichletBC, \
    TrialFunction, TestFunction, solve, plot, lhs, rhs, grad, dot, dx, ds, assemble, interpolate, \
    BoundaryMesh

x_left, x_coupling = 0, 1
y_bottom, y_top = 0, 1
p0 = Point(x_left, y_bottom)
p1 = Point(x_coupling, y_top)
nx, ny = 20, 20
alpha = 3.0  # parameter alpha
beta = 1.0  # parameter beta
dt = 1

u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
f = Constant(beta - 2 - 2 * alpha)
f_N = Expression('2 * x[0]', degree=1)
mesh = RectangleMesh(p0, p1, nx, ny, diagonal='crossed')
V = FunctionSpace(mesh, 'P', 1)
S = FunctionSpace(BoundaryMesh(mesh, 'exterior'), 'DG', 1)
u_n = interpolate(u_D, V)
u_D.t = dt

v = TestFunction(V)
w = TestFunction(V)

u = TrialFunction(V)
u_np1 = Function(V)


def pde(u, v):
    dudt = (u - u_n) / dt
    F = 0
    F += dot(grad(u), grad(v)) * dx
    F += dudt * v * dx
    F += -f * v * dx
    return F


bcStr = "near(x[0]," + str(x_left) + ")" \
        " || " \
        "near(x[0]," + str(x_coupling) + ")" \
        " || " \
        "near(x[1]," + str(y_bottom) + ")" \
        " || " \
        "near(x[1]," + str(y_top) + ")"

bcStr = "("+bcStr+")"
bc = DirichletBC(V, u_D, bcStr)
res = pde(TrialFunction(V), v)
solve(lhs(res) == rhs(res), u_np1, bc)

flux_res = pde(u_np1, v) - TrialFunction(V) * v * ds
q = Function(V)
ABdry = assemble(lhs(flux_res), keep_diagonal=True)
bBdry = assemble(rhs(flux_res))
qBC = DirichletBC(V, Constant(0.0), "!"+bcStr)
qBC.apply(ABdry, bBdry)
solve(ABdry, q.vector(), bBdry)
q.set_allow_extrapolation(True)
fluxes = interpolate(q, S)

dudt = (u_np1 - u_n) / dt

print("Flux: "+str(assemble(fluxes * dx)))
print("Source term: "+str(assemble(f * dx(domain=mesh) - dudt * dx())))

import numpy as np
from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(np.linspace(0, 1), [fluxes(x_left, y) for y in np.linspace(0, 1)], label="left edge")
plt.plot(np.linspace(1, 2), [fluxes(x, y_top) for x in np.linspace(0, 1)], label="top edge")
plt.plot(np.linspace(2, 3), [fluxes(x_coupling, y) for y in np.linspace(1, 0)], label="right edge")
plt.plot(np.linspace(3, 4), [fluxes(x, y_bottom) for x in np.linspace(1, 0)], label="bottom edge")
plt.legend()
plt.ylabel('normal heat flux')
plt.xlabel('arc length')

plt.figure(2)
plot(u_np1)
x, y = map(list, zip(*[(x_left - .1 * fluxes(x_left, y), y) for y in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_top + .1 * fluxes(x, y_top)) for x in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x_coupling + .1 * fluxes(x_coupling, y), y) for y in np.linspace(1, 0)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_bottom - .1 * fluxes(x, y_bottom)) for x in np.linspace(1, 0)]))
plt.plot(x, y)
plt.show()