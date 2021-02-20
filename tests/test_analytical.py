from firedrake import *
from firedrake_adjoint import *
from pyMMAopt import MMASolver, ReducedInequality

def test_analytical():
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    DG = VectorFunctionSpace(mesh, "DG", 0)
    X = Function(DG)
    with stop_annotating():
        X.interpolate(Constant((1.234, 2.345)))
    x, y = split(X)
    J = assemble(sqrt(y) * dx)
    G1 = assemble(((2.0 * x) ** 3 - y) * dx)
    G2 = assemble(((-1.0 * x + 1.0) ** 3 - y) * dx)

    m = Control(X)

    Jhat = ReducedFunctional(J, m)
    G1hat = ReducedFunctional(G1, m)
    G2hat = ReducedFunctional(G2, m)
    print(f"{G1hat.derivative().dat.data_ro}")
    print(f"{G2hat.derivative().dat.data_ro}")

    problem = MinimizationProblem(
        Jhat,
        bounds=(0.0, 100.0),
        constraints=[
            ReducedInequality(G1hat, 1e-5, Control(G1), normalized=False),
            ReducedInequality(G2hat, 1e-5, Control(G2), normalized=False),
        ],
    )

    parameters_mma = {
        "move": 0.1,
        "maximum_iterations": 10,
        "m": 2,
        "IP": 0,
        "tol": 1e-9,
        "accepted_tol": 1e-8,
        "gcmma": True,
    }
    solver = MMASolver(problem, parameters=parameters_mma)
    rho_opt = solver.solve()
    assert abs(Jhat(rho_opt) - 0.5443418101973394) < 1e-7
    solution = Function(DG).interpolate(Constant((0.33332924, 0.29630801)))
    assert errornorm(rho_opt, solution) < 1e-7