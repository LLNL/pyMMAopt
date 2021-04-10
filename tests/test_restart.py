from firedrake import *
from firedrake_adjoint import *
from pyMMAopt import MMASolver, ReducedInequality

import os, signal, itertools


def test_save_with_signal():
    mesh = UnitSquareMesh(10, 10)
    DG = FunctionSpace(mesh, "DG", 0)
    print(f"DOFS: {DG.dim()}")
    rho = interpolate(Constant(1.0), DG)
    J = assemble(rho * rho * rho * dx)
    G = assemble(rho * dx)
    m = Control(rho)

    g_counter = itertools.count()

    def deriv_cb(j, dj, rho):
        iter = next(g_counter)
        if iter % 10 == 0 and iter > 0:
            os.kill(os.getpid(), signal.SIGUSR1)

    Jhat = ReducedFunctional(J, m, derivative_cb_post=deriv_cb)
    Ghat = ReducedFunctional(G, m)
    total_area = assemble(Constant(1.0) * dx(domain=mesh), annotate=False)
    Glimit = total_area * 50
    Gcontrol = Control(G)

    problem = MinimizationProblem(
        Jhat,
        bounds=(1e-5, 100.0),
        constraints=[
            ReducedInequality(Ghat, Glimit, Gcontrol),
        ],
    )

    parameters_mma = {
        "move": 0.1,
        "maximum_iterations": 10,
        "m": 1,
        "IP": 0,
        "tol": 1e-9,
        "accepted_tol": 1e-8,
        "norm": "L2",
    }
    solver = MMASolver(problem, parameters=parameters_mma)
    rho_sol = solver.solve()

    assert os.path.isfile("checkpoint.h5")

    parameters_mma["restart_file"] = "./checkpoint.h5"
    parameters_mma["maximum_iterations"] = 0
    solver = MMASolver(problem, parameters=parameters_mma)
    rho_restart = solver.solve()
    assert errornorm(rho_sol, rho_restart) < 1e-2

