from firedrake import *
from firedrake_adjoint import *
import pytest

from pyMMAopt import MMASolver, ReducedInequality


@pytest.mark.parametrize(
    "norm,result",
    [["l2", 7.454771069410802], ["L2", 7.420380654729631]],
)
def test_compliance(norm, result):
    mesh = RectangleMesh(100, 30, 10, 3)

    V = VectorFunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    print(f"# DOFS: {V.dim()}")

    # Elasticity parameters
    E, nu = 1e0, 0.3
    mu, lmbda = Constant(E / (2 * (1 + nu))), Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )

    # Helmholtz solver
    RHO = FunctionSpace(mesh, "DG", 0)
    rho = interpolate(Constant(0.1), RHO)
    af, b = TrialFunction(RHO), TestFunction(RHO)

    filter_radius = Constant(0.02)
    x, y = SpatialCoordinate(mesh)
    x_ = interpolate(x, RHO)
    y_ = interpolate(y, RHO)
    Delta_h = sqrt(jump(x_) ** 2 + jump(y_) ** 2)

    rhof = Function(RHO)
    solver_params = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 200,
        "mat_mumps_icntl_24": 1,
    }

    eps = Constant(1e-5)
    p = Constant(3.0)

    def simp(rho):
        return eps + (Constant(1.0) - eps) * rho ** p

    def epsilon(v):
        return sym(nabla_grad(v))

    def sigma(v):
        return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(2)

    DIRICHLET = 1
    NEUMANN = 2
    load = Constant((0.0, -5.0))

    c = Control(rho)

    def forward(rho):

        aH = filter_radius * jump(af) / Delta_h * jump(b) * dS + af * b * dx
        LH = rho * b * dx

        solve(aH == LH, rhof, solver_parameters=solver_params)
        rhofControl = Control(rhof)

        a = inner(simp(rhof) * sigma(u), epsilon(v)) * dx
        L = inner(load, v) * ds(NEUMANN)

        u_sol = Function(V)

        bcs = DirichletBC(V, Constant((0.0, 0.0)), DIRICHLET)

        solve(a == L, u_sol, bcs=bcs, solver_parameters=solver_params)

        return rhof, u_sol

    rhof, u_sol = forward(rho)
    solution_pvd = File("compliance_design.pvd")
    rho_viz = Function(RHO)

    def deriv_cb(j, dj, rho):
        with stop_annotating():
            rho_viz.assign(rho)
            solution_pvd.write(rho_viz)

    J = assemble(Constant(1e-4) * inner(u_sol, load) * ds(NEUMANN))
    Vol = assemble(rhof * dx)
    VolControl = Control(Vol)

    with stop_annotating():
        Vlimit = assemble(Constant(1.0) * dx(domain=mesh)) * 0.5

    Jhat = ReducedFunctional(J, c, derivative_cb_post=deriv_cb)
    Volhat = ReducedFunctional(Vol, c)

    lb = 0.0
    ub = 1.0
    problem = MinimizationProblem(
        Jhat,
        bounds=(lb, ub),
        constraints=[ReducedInequality(Volhat, Vlimit, VolControl)],
    )

    parameters_mma = {
        "move": 0.2,
        "maximum_iterations": 20,
        "m": 1,
        "IP": 0,
        "tol": 1e-6,
        "accepted_tol": 1e-4,
        "gcmma": True,
        "norm": norm,
    }
    solver = MMASolver(problem, parameters=parameters_mma)

    results = solver.solve()
    rho_opt = results["control"]

    final_cost_func = Jhat(rho_opt)

    assert np.allclose(final_cost_func, result, rtol=1e-5)


if __name__ == "__main__":
    test_compliance("L2", 7.420380654729631)
