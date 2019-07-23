# extremely helpful: https://fenicsproject.discourse.group/t/project-gradient-on-boundarymesh/262

from fenics import Function, SubDomain, RectangleMesh, FunctionSpace, Point, Expression, Constant, DirichletBC, \
    TrialFunction, TestFunction, File, solve, plot, lhs, rhs, grad, inner, dot, dx, ds, assemble, interpolate, \
    project, near, VectorFunctionSpace, BoundaryMesh, Measure, FacetNormal, FiniteElement, VectorElement, TrialFunctions, TestFunctions

from matplotlib import pyplot as plt

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
f_N = Expression(('2 * x[0]', '2 * alpha * x[1]'), alpha=alpha, degree=1)
mesh = RectangleMesh(p0, p1, nx, ny, diagonal='crossed')
P = FiniteElement("Lagrange", mesh.ufl_cell(), 2)  # use quadratic temperature triangle
Q = VectorElement("Lagrange", mesh.ufl_cell(), 1)  # use linear flux
V = FunctionSpace(mesh, P * Q)

u_n = interpolate(u_D, V.sub(0).collapse())
q_n = interpolate(f_N, V.sub(1).collapse())
u_D.t = dt

(v, w) = TestFunctions(V)
(u, q) = TrialFunctions(V)
mixed_np1 = Function(V)
u_np1, q_np1 = mixed_np1.split()


def pde(u, q, v, w):
    dudt = (u - u_n) / dt
    F = 0
    F += dot(q, grad(v)) * dx
    F += dudt * v * dx
    F += -f * v * dx
    F += dot(grad(u), w) * dx
    F += -dot(q, w) * dx
    return F


bcStr = ""


bcStr = "near(x[0]," + str(x_left) + ")" \
        " || " \
        "near(x[0]," + str(x_coupling) + ")" \
        " || " \
        "near(x[1]," + str(y_bottom) + ")" \
        " || " \
        "near(x[1]," + str(y_top) + ")"

bcStr = "("+bcStr+")"
bc = DirichletBC(V.sub(0), u_D, bcStr)
F = pde(u, q, v, w)
solve(lhs(F) == rhs(F), mixed_np1, bc)

dudt = (u_np1 - u_n) / dt

normal = FacetNormal(mesh)

print("Flux: "+str(assemble(dot(q_np1, normal) * ds)))
print("Source term: "+str(assemble((f - dudt) * dx(domain=mesh))))

import numpy as np
from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(np.linspace(0, 1), [q_np1(x_left, y)[0] for y in np.linspace(0, 1)])
plt.plot(np.linspace(1, 2), [q_np1(x, y_top)[1] for x in np.linspace(0, 1)])
plt.plot(np.linspace(2, 3), [q_np1(x_coupling, y)[0] for y in np.linspace(1, 0)])
plt.plot(np.linspace(3, 4), [q_np1(x, y_bottom)[1] for x in np.linspace(1, 0)])

plt.figure(2)
plot(u_np1)
x, y = map(list, zip(*[(x_left - .1 * q_np1(x_left, y)[0], y) for y in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_top + .1 * q_np1(x, y_top)[1]) for x in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x_coupling + .1 * q_np1(x_coupling, y)[0], y) for y in np.linspace(1, 0)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_bottom - .1 * q_np1(x, y_bottom)[1]) for x in np.linspace(1, 0)]))
plt.plot(x, y)

plt.show()
