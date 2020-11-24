from firedrake import *
from firedrake_adjoint import *
from pyMMAopt import MMASolver

mesh = UnitSquareMesh(50, 50)
DG = FunctionSpace(mesh, 'DG', 0)
x, y = SpatialCoordinate(mesh)
rho = interpolate(x**2*y**3, DG)

controls_f = File("design.pvd")
gradient_f = File("gradient.pvd")
rho_viz = Function(DG)
grad_viz = Function(DG)
def deriv_cb(j, dj, rho):
    rho_viz.assign(rho)
    controls_f.write(rho_viz)
    grad_viz.assign(dj)
    gradient_f.write(grad_viz)

J = assemble(Constant(1e4)*rho*rho*dx)
G = assemble(rho*dx)
m = Control(rho)
Jhat = ReducedFunctional(J, m, derivative_cb_post=deriv_cb)
Ghat = ReducedFunctional(G, m)
Glimit = 5.0
Gcontrol = Control(G)

class ReducedInequality(InequalityConstraint):
    def __init__(self, Ghat, Glimit, Gcontrol):
        self.Ghat = Ghat
        self.Glimit = float(Glimit)
        self.Gcontrol = Gcontrol

    def function(self, m):
        # Compute the integral of the control over the domain
        integral = self.Gcontrol.tape_value()
        print(f"Constraint function: {integral}, Constraint upper bound {self.Glimit}")
        with stop_annotating():
            value = -integral / self.Glimit + 1.0
        return [value]

    def jacobian(self, m):
        with stop_annotating():
            gradients = self.Ghat.derivative()
            with gradients.dat.vec as v:
                v.scale(-1.0 / self.Glimit)
        return [gradients]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1

problem = MinimizationProblem(
    Jhat,
    bounds=(1e-6, 1.0),
    constraints=[
        ReducedInequality(Ghat, Glimit, Gcontrol),
    ],
)


parameters_mma = {
    "move": 0.1,
    "maximum_iterations": 200,
    "m": 1,
    "IP": 0,
    "tol": 1e-7,
    "accepted_tol": 1e-5,
}
solver = MMASolver(problem, parameters=parameters_mma)
solver.solve()
