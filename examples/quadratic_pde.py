from firedrake import *
from firedrake.petsc import PETSc
from firedrake_adjoint import *
from pyMMAopt import MMASolver
import os

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
    default=200,
)
args = parser.parse_args()
n_vars = args.n_vars
grid_resol = int(np.sqrt(n_vars))


mesh = Mesh("./corner_mesh.msh")
DG = FunctionSpace(mesh, "DG", 0)
x, y = SpatialCoordinate(mesh)
rho = interpolate(Constant(1.0), DG)

solution_pvd = File("quadratic_sol.pvd")
derivative_pvd = File("derivative.pvd")
rho_viz = Function(DG)
der_viz = Function(DG)


def deriv_cb(j, dj, rho):
    with stop_annotating():
        rho_viz.assign(rho)
        solution_pvd.write(rho_viz)

        der_viz.assign(dj)
        derivative_pvd.write(der_viz)


J = assemble(Constant(1e4) * rho * rho * dx)
G = assemble(rho * dx)
m = Control(rho)
Jhat = ReducedFunctional(J, m, derivative_cb_post=deriv_cb)
Ghat = ReducedFunctional(G, m)
total_area = assemble(Constant(1.0) * dx(domain=mesh), annotate=False)
Glimit = total_area * 20.0
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
    bounds=(1e-5, 100.0),
    constraints=[
        ReducedInequality(Ghat, Glimit, Gcontrol),
    ],
)


parameters_mma = {
    "move": 0.1,
    "maximum_iterations": 5,
    "m": 1,
    "IP": 0,
    "tol": 1e-9,
    "accepted_tol": 1e-8,
    "norm": "L2",
}
solver = MMASolver(problem, parameters=parameters_mma)
rho_sol = solver.solve()
