from firedrake import *
from firedrake.petsc import PETSc
from firedrake_adjoint import *
from pyMMAopt import MMASolver

print = lambda x: PETSc.Sys.Print(x, comm=COMM_SELF)

import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Simple optimization problem")
parser.add_argument(
    "--n_vars",
    action="store",
    dest="n_vars",
    type=int,
    help="Number of design variables",
    default=100,
)
args = parser.parse_args()
n_vars = args.n_vars
grid_resol = int(np.sqrt(n_vars))


mesh = UnitSquareMesh(grid_resol, grid_resol)
DG = FunctionSpace(mesh, 'DG', 0)
x, y = SpatialCoordinate(mesh)
rho = interpolate(x ** 2 * y ** 3, DG)

J = assemble(Constant(1e4)*rho*rho*dx)
G = assemble(rho*dx)
m = Control(rho)
Jhat = ReducedFunctional(J, m)
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
    "maximum_iterations": 3,
    "m": 1,
    "IP": 0,
    "tol": 1e-7,
    "accepted_tol": 1e-5,
}
solver = MMASolver(problem, parameters=parameters_mma)
solver.solve()
