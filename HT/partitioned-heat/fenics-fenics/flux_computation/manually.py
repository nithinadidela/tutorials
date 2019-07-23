# from http://hplgit.github.io/INF5620/doc/pub/fenics_tutorial1.1/tu2.html#tut-poisson-gradu


from fenics import Function, RectangleMesh, FunctionSpace, Point, Expression, Constant, DirichletBC, \
    TrialFunction, TestFunction, solve, plot, lhs, rhs, grad, inner, dot, dx, ds, assemble, interpolate, \
    VectorFunctionSpace


def fluxes_from_temperature_full_domain(F, V):
    """
    compute flux from weak form (see p.3 in Toselli, Andrea, and Olof Widlund. Domain decomposition methods-algorithms and theory. Vol. 34. Springer Science & Business Media, 2006.)
    :param F: weak form with known u^{n+1}
    :param V: function space
    :param hy: spatial resolution perpendicular to flux direction
    :return:
    """
    fluxes_vector = assemble(F)  # assemble weak form -> evaluate integral
    v = TestFunction(V)
    fluxes = Function(V)  # create function for flux
    area = assemble(v * ds).get_local()
    for i in range(area.shape[0]):
        if area[i] != 0:  # put weight from assemble on function
            fluxes.vector()[i] = fluxes_vector[i] / area[i]  # scale by surface area
        else:
            assert(abs(fluxes_vector[i]) < 10**-10)  # for non surface parts, we expect zero flux
            fluxes.vector()[i] = fluxes_vector[i]
    return fluxes


x_left, x_coupling = 0, 1
y_bottom, y_top = 0, 1
p0 = Point(x_left, y_bottom)
p1 = Point(x_coupling, y_top)
nx, ny = 40, 40
alpha = 3.0  # parameter alpha
beta = 1.3  # parameter beta
dt = 1

u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
f = Constant(beta - 2 - 2 * alpha)
f_N = Expression('2 * x[0]', degree=1)
mesh = RectangleMesh(p0, p1, nx, ny, diagonal='crossed')
V = FunctionSpace(mesh, 'P', 1)
V_flux = VectorFunctionSpace(mesh, 'P', 1)
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

F_known_u = u_np1 * v / dt * dx + dot(grad(u_np1), grad(v)) * dx - (u_n / dt + f) * v * dx

fluxes = fluxes_from_temperature_full_domain(F_known_u, V)

dudt = (u_np1 - u_n) / dt

print("Flux: "+str(assemble(fluxes * ds)))
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