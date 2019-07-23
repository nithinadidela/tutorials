# from http://hplgit.github.io/INF5620/doc/pub/fenics_tutorial1.1/tu2.html#tut-poisson-gradu


from fenics import Function, SubDomain, RectangleMesh, FunctionSpace, Point, Expression, Constant, DirichletBC, \
    TrialFunction, TestFunction, solve, plot, lhs, rhs, grad, inner, dot, dx, ds, assemble, interpolate, \
    near, VectorFunctionSpace, Measure, MeshFunction

x_left, x_coupling = 0, 1
y_bottom, y_top = 0, 1
p0 = Point(x_left, y_bottom)
p1 = Point(x_coupling, y_top)
nx, ny = 4, 4
alpha = 3.0  # parameter alpha
beta = 1.3  # parameter beta
dt = 1

u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
f = Constant(beta - 2 - 2 * alpha)
f_N = Expression('2 * x[0]', degree=1)
mesh = RectangleMesh(p0, p1, nx, ny, diagonal='crossed')
V = FunctionSpace(mesh, 'P', 2)
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
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)


class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (near(x[0], x_left, tol) or near(x[0], x_coupling, tol))


class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (near(x[1], y_bottom, tol) or near(x[1], y_top, tol))


bx0 = BoundaryX0()
bx0.mark(boundary_markers, 0)
bx1 = BoundaryX1()
bx1.mark(boundary_markers, 1)
bc = DirichletBC(V, u_D, bcStr)
res = pde(TrialFunction(V), v)
solve(lhs(res) == rhs(res), u_np1, bc)

V_g = VectorFunctionSpace(mesh, 'Lagrange', 1)
w = TrialFunction(V_g)
v = TestFunction(V_g)

a = inner(w, v)*dx
L = inner(grad(u_np1), v)*dx
grad_u = Function(V_g)
solve(a == L, grad_u)

flux_x, flux_y = grad_u.split(deepcopy=True)  # extract components
dudt = (u_np1 - u_n) / dt

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

print("Flux: "+str(assemble(flux_x * ds(0) + flux_y * ds(1))))
print("Source term: "+str(assemble(f * dx(domain=mesh) - dudt * dx())))

import numpy as np
from matplotlib import pyplot as plt
plt.figure(1)
plt.plot(np.linspace(0, 1), [flux_x(x_left, y) for y in np.linspace(0, 1)], label="left edge")
plt.plot(np.linspace(1, 2), [flux_y(x, y_top) for x in np.linspace(0, 1)], label="top edge")
plt.plot(np.linspace(2, 3), [flux_x(x_coupling, y) for y in np.linspace(1, 0)], label="right edge")
plt.plot(np.linspace(3, 4), [flux_y(x, y_bottom) for x in np.linspace(1, 0)], label="bottom edge")
plt.legend()
plt.ylabel('normal heat flux')
plt.xlabel('arc length')

plt.figure(2)
plot(u_np1)
x, y = map(list, zip(*[(x_left - .1 * flux_x(x_left, y), y) for y in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_top + .1 * flux_y(x, y_top)) for x in np.linspace(0, 1)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x_coupling + .1 * flux_x(x_coupling, y), y) for y in np.linspace(1, 0)]))
plt.plot(x, y)
x, y = map(list, zip(*[(x, y_bottom - .1 * flux_y(x, y_bottom)) for x in np.linspace(1, 0)]))
plt.plot(x, y)
plt.show()