import firedrake as fd
from firedrake import sqrt, jump, dx, ds, dS, inner, sym, nabla_grad, tr, Identity
import firedrake_adjoint as fda

from pyMMAopt import MMASolver
import argparse

def compliance():
    parser = argparse.ArgumentParser(description="Compliance problem with MMA")
    parser.add_argument(
        "--nref",
        action="store",
        dest="nref",
        type=int,
        help="Number of mesh refinements",
        default=2,
    )
    parser.add_argument(
        "--uniform",
        action="store",
        dest="uniform",
        type=int,
        help="Use uniform mesh",
        default=0,
    )
    parser.add_argument(
        "--inner_product",
        action="store",
        dest="inner_product",
        type=str,
        help="Inner product, euclidean or L2",
        default="L2",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        type=str,
        help="Directory for all the output",
        default="./",
    )
    args = parser.parse_args()
    nref = args.nref
    inner_product = args.inner_product
    output_dir = args.output_dir

    assert inner_product == "L2" or inner_product == "euclidean"

    mesh = fd.Mesh("./beam_uniform.msh")

    if nref > 0:
        mh = fd.MeshHierarchy(mesh, nref)
        mesh = mh[-1]
    elif nref < 0:
        raise RuntimeError("Non valid mesh argument")

    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    u, v = fd.TrialFunction(V), fd.TestFunction(V)

    # Elasticity parameters
    E, nu = 1e0, 0.3
    mu, lmbda = fd.Constant(E / (2 * (1 + nu))), fd.Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))

    # Helmholtz solver
    RHO = fd.FunctionSpace(mesh, "DG", 0)
    rho = fd.interpolate(fd.Constant(0.1), RHO)
    af, b = fd.TrialFunction(RHO), fd.TestFunction(RHO)

    filter_radius = fd.Constant(0.2)
    x, y = fd.SpatialCoordinate(mesh)
    x_ = fd.interpolate(x, RHO)
    y_ = fd.interpolate(y, RHO)
    Delta_h = sqrt(jump(x_) ** 2 + jump(y_) ** 2)
    aH = filter_radius * jump(af) / Delta_h * jump(b) * dS + af * b * dx
    LH = rho * b * dx

    rhof = fd.Function(RHO)
    solver_params = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": 200,
        "mat_mumps_icntl_24": 1,
    }
    fd.solve(aH == LH, rhof, solver_parameters=solver_params)
    rhofControl = fda.Control(rhof)

    eps = fd.Constant(1e-5)
    p = fd.Constant(3.0)


    def simp(rho):
        return eps + (fd.Constant(1.0) - eps) * rho ** p


    def epsilon(v):
        return sym(nabla_grad(v))


    def sigma(v):
        return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(2)


    DIRICHLET = 3
    NEUMANN = 4

    a = inner(simp(rhof) * sigma(u), epsilon(v)) * dx
    load = fd.Constant((0.0, -1.0))
    L = inner(load, v) * ds(NEUMANN)

    u_sol = fd.Function(V)


    bcs = fd.DirichletBC(V, fd.Constant((0.0, 0.0)), DIRICHLET)

    fd.solve(a == L, u_sol, bcs=bcs, solver_parameters=solver_params)
    c = fda.Control(rho)
    J = fd.assemble(fd.Constant(1e-4) * inner(u_sol, load) * ds(NEUMANN))
    Vol = fd.assemble(rhof * dx)
    VolControl = fda.Control(Vol)


    with fda.stop_annotating():
        Vlimit = fd.assemble(fd.Constant(1.0) * dx(domain=mesh)) * 0.3

    rho_viz_f = fd.Function(RHO, name="rho")
    plot_file = f"{output_dir}/design_{inner_product}.pvd"
    print(plot_file)
    controls_f = fd.File(plot_file)


    def deriv_cb(j, dj, rho):
        with fda.stop_annotating():
            rho_viz_f.assign(rhofControl.tape_value())
            controls_f.write(rho_viz_f)


    Jhat = fda.ReducedFunctional(J, c, derivative_cb_post=deriv_cb)
    Volhat = fda.ReducedFunctional(Vol, c)


    class VolumeConstraint(fda.InequalityConstraint):
        def __init__(self, Vhat, Vlimit, VolControl):
            self.Vhat = Vhat
            self.Vlimit = float(Vlimit)
            self.VolControl = VolControl

        def function(self, m):
            # Compute the integral of the control over the domain
            integral = self.VolControl.tape_value()
            with fda.stop_annotating():
                value = -integral / self.Vlimit + 1.0
            return [value]

        def jacobian(self, m):
            with fda.stop_annotating():
                gradients = self.Vhat.derivative()
                with gradients.dat.vec as v:
                    v.scale(-1.0 / self.Vlimit)
            return [gradients]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1


    lb = 1e-5
    ub = 1.0
    problem = fda.MinimizationProblem(
        Jhat, bounds=(lb, ub), constraints=[VolumeConstraint(Volhat, Vlimit, VolControl)]
    )

    parameters_mma = {
        "move": 0.2,
        "maximum_iterations": 200,
        "m": 1,
        "IP": 0,
        "tol": 1e-6,
        "accepted_tol": 1e-4,
        "norm": inner_product,
        "gcmma": True,
    }
    solver = MMASolver(problem, parameters=parameters_mma)

    rho_opt = solver.solve()

    with open(f"{output_dir}/finished_{inner_product}.txt", "w") as f:
        f.write("Done")

if __name__ == "__main__":
    compliance()