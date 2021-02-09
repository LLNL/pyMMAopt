# from fenics_adjoint import *
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import gather
from firedrake import (
    PETSc,
    Function,
    assemble,
    dx,
    TrialFunction,
    TestFunction,
    Constant,
)
from firedrake import COMM_SELF, HDF5File
from mpi4py import MPI
import time
import signal


try:
    from .mma import MMAClient
except ImportError:
    print("You need to install MMA")
    raise
import numpy


print = lambda x: PETSc.Sys.Print(x, comm=COMM_SELF)


class MMASolver(OptimizationSolver):
    def __init__(self, problem, parameters=None):
        OptimizationSolver.__init__(self, problem, parameters)

        self.rf = self.problem.reduced_functional
        if len(self.rf.controls) > 1:
            raise RuntimeError("Only one control is possible for MMA")

        if isinstance(self.rf.controls[0].control, Function) is False:
            raise RuntimeError("Only control of type Function is possible for MMA")

        control_funcspace = self.rf.controls[0].control.function_space()
        control_elem = control_funcspace.ufl_element()

        supported_fe = ["DQ", "Discontinuous Lagrange"]
        if control_elem.family() == "TensorProductElement":
            sub_elem = control_elem.sub_elements()
            if (
                sub_elem[0].family() not in supported_fe
                or sub_elem[0].degree() != 0
                or sub_elem[1].family() not in supported_fe
                or sub_elem[1].degree() != 0
            ):
                raise RuntimeError(
                    "Only zero degree Discontinuous Galerkin function space for extruded elements is supported"
                )
        else:
            if control_elem.family() not in supported_fe or control_elem.degree() != 0:
                raise RuntimeError(
                    "Only zero degree Discontinuous Galerkin function space is supported"
                )

        if parameters.get("norm") == "L2":
            self.Mdiag = assemble(
                TrialFunction(control_funcspace) * TestFunction(control_funcspace) * dx,
                diagonal=True,
            ).dat.data_ro
        else:
            self.Mdiag = numpy.ones(len(self.rf.controls[0].control.dat.data_ro))

        self.__build_mma_problem()
        self.__set_parameters()

        self.change = 0.0
        self.f0val = 0.0
        self.g0val = [[0.0] for _ in range(self.m)]
        self.loop = 0

    def __set_parameters(self):
        """Set some basic parameters from the parameters dictionary that the user
        passed in, if any."""
        param_defaults = {
            "m": 1,
            "tol": 1e-8,
            "accepted_tol": 1e-4,
            "maximum_iterations": 100,
            "asyinit": 0.5,
            "asyincr": 1.2,
            "asydecr": 0.7,
            "albefa": 0.1,
            "move": 0.1,
            "epsimin": 1.0e-05,
            "raa0": 1.0e-05,
            "xmin": [],
            "xmax": [],
            "a0": 1.0,
            "a": [],
            "c": [],
            "d": [],
            "IP": 0,
            "norm": "l2",
        }
        if self.parameters is not None:
            for key in self.parameters.keys():
                if key not in param_defaults.keys():
                    raise ValueError(
                        "Don't know how to deal with parameter %s (a %s)"
                        % (key, self.parameters[key].__class__)
                    )

            for (prop, default) in param_defaults.items():
                self.parameters[prop] = self.parameters.get(prop, default)
        else:
            self.parameters = param_defaults

    def __build_mma_problem(self):
        """Build the pyipopt problem from the OptimizationProblem instance."""

        self.rf = self.problem.reduced_functional
        assert len(self.rf.controls) == 1, "Only one control is possible for MMA"
        assert isinstance(
            self.rf.controls[0].control, Function
        ), "Only control of type Function is possible for MMA"
        control = self.rf.controls[0]
        n_local_control = len(
            control.control.dat.data_ro
        )  # There is another function: dat.data_ro_with_halos (might be useful)

        (self.lb, self.ub) = self.__get_bounds()
        (nconstraints, self.fun_g, self.jac_g) = self.__get_constraints()

        self.n = n_local_control
        self.m = nconstraints
        # if isinstance(self.problem, MaximizationProblem):
        # multiply objective function by -1 internally in
        # ipopt to maximise instead of minimise
        # nlp.num_option('obj_scaling_factor', -1.0)

    def __get_bounds(self):
        r"""Convert the bounds into the format accepted by MMA (two numpy arrays,
        one for the lower bound and one for the upper)."""

        bounds = self.problem.bounds

        if bounds is not None:
            lb_list = []
            ub_list = []  # a list of numpy arrays, one for each control

            for (bound, control) in zip(bounds, self.rf.controls):
                general_lb, general_ub = bound  # could be float, Constant, or Function

                if isinstance(control.control, Function):
                    n_local_control = len(
                        control.control.dat.data_ro
                    )  # There is another function: dat.data_ro_with_halos (might be useful)
                elif isinstance(control.control, Constant) or isinstance(
                    control.control, AdjFloat
                ):
                    n_local_control = 1
                else:
                    raise TypeError(
                        f"Type of control: {type(control.control)} not supported by pyMMAopt"
                    )

                if isinstance(general_lb, (float, int)):
                    lb = numpy.array([float(general_lb)] * n_local_control)
                else:
                    with general_lb.dat.vec_ro as lb_v:
                        lb = lb_v.array

                lb_list.append(lb)

                if isinstance(general_ub, (float, int)):
                    ub = numpy.array([float(general_ub)] * n_local_control)
                else:
                    with general_ub.dat.vec_ro as ub_v:
                        ub = ub_v.array

                ub_list.append(ub)

            ub = numpy.concatenate(ub_list)
            lb = numpy.concatenate(lb_list)

        else:
            # Unfortunately you really need to specify bounds, I think?!
            ncontrols = len(self.rf.get_controls())
            max_float = numpy.finfo(numpy.double).max
            ub = numpy.array([max_float] * ncontrols)

            min_float = numpy.finfo(numpy.double).min
            lb = numpy.array([min_float] * ncontrols)

        return (lb, ub)

    def __get_constraints(self):
        constraint = self.problem.constraints

        if constraint is None:
            # The length of the constraint vector
            nconstraints = 0

            # The bounds for the constraint
            empty = numpy.array([], dtype=float)

            # The constraint function, should do nothing
            def fun_g(x, user_data=None):
                return empty

            # The constraint Jacobian
            def jac_g(x, flag, user_data=None):
                if flag:
                    rows = numpy.array([], dtype=int)
                    cols = numpy.array([], dtype=int)
                    return (rows, cols)
                else:
                    return empty

            return (nconstraints, fun_g, jac_g)
        else:
            # The length of the constraint vector
            nconstraints = constraint._get_constraint_dim()
            # The constraint function
            def fun_g(x, user_data=None):
                out = numpy.array(constraint.function(x), dtype=float)
                return out

            # The constraint Jacobian:
            # flag = True  means 'tell me the sparsity pattern';
            # flag = False means 'give me the damn Jacobian'.
            def jac_g(x, user_data=None):
                out = constraint.jacobian(x)
                return out

            return (nconstraints, fun_g, jac_g)

    def solve(self):

        change = 1.0
        loop = 1
        tol = self.parameters["tol"]
        accepted_tol = self.parameters["accepted_tol"]
        # Initial estimation
        control_function = self.rf.controls[0].control
        with control_function.dat.vec_ro as control_vec:
            a_np = control_vec.array

        def receive_signal(signum, stack):
            print(f"SIGNAL RECEIVED!!!!!!!!!!!!!!")
            with HDF5File(output_dir + "/final_design", "w") as checkpoint:
                checkpoint.write(hs_rho_viz, "/final_design")

        signal.signal(signal.SIGUSR1, receive_signal)

        import numpy as np

        self.parameters["xmin"] = self.lb
        self.parameters["xmax"] = self.ub
        self.parameters["n"] = control_function.function_space().dim()
        self.parameters["Mdiag"] = self.Mdiag
        self.parameters["gcmma"] = True
        itermax = self.parameters["maximum_iterations"]

        # Create an optimizer client
        clientOpt = MMAClient(self.parameters)
        #'asyinit':0.2,'asyincr':0.8,'asydecr':0.3

        xold1 = np.copy(a_np)
        xold2 = np.copy(a_np)
        low = np.array([])
        upp = np.array([])

        change_arr = []

        a_function = control_function.copy(deepcopy=True)

        def eval_f(a_np):
            with a_function.dat.vec as a_vec:
                a_vec.array_w = a_np
            f0val = self.rf(a_function)
            return f0val

        def eval_g(a_np):
            with a_function.dat.vec as a_vec:
                a_vec.array_w = a_np
            return -1.0 * self.fun_g(a_function)

        dg0dx = np.empty([self.m, self.n])
        df0dx = np.empty([self.n])
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        while change > tol and loop <= itermax:
            t0 = time.time()
            # Cost functions
            f0val = eval_f(a_np)
            g0val = eval_g(a_np)

            # Gradients
            df0dx_func = self.rf.derivative()
            jac = self.jac_g(a_function)

            # Copy into the numpy arrays
            with df0dx_func.dat.vec_ro as df_vec:
                df0dx[:] = df_vec.array
            for j, jac_j in enumerate(jac):
                with jac_j[0].dat.vec_ro as jac_vec:
                    dg0dx[j, :] = -1.0 * jac_vec.array

            # move limits
            clientOpt.xmin = self.lb
            clientOpt.xmax = self.ub

            xmma, y, z, lam, xsi, eta, mu, zet, s, low, upp, factor = clientOpt.mma(
                a_np,
                xold1,
                xold2,
                low,
                upp,
                f0val,
                g0val,
                df0dx,
                dg0dx,
                loop,
                eval_f=eval_f,
                eval_g=eval_g,
            )

            kkt_norm = clientOpt.residualKKTPrimal(
                xmma,
                y,
                z,
                lam,
                df0dx,
                g0val,
                dg0dx,
            )

            local_change = np.abs(np.max(xmma - xold1))
            change = comm.allreduce(local_change, op=MPI.MAX)
            # update design variables
            xold2 = np.copy(xold1)
            xold1 = np.copy(a_np)
            a_np = np.copy(xmma)
            loop = loop + 1

            PETSc.Sys.Print("It: {it}, obj: {obj} ".format(it=loop, obj=f0val), end="")
            PETSc.Sys.Print(
                *(map("g[{0[0]}]: {0[1][0]} ".format, enumerate(g0val))), end=""
            )
            # PETSc.Sys.Print(" Inner iterations: {:d}".format(inner_it), end="")
            PETSc.Sys.Print(" kkt: {:6f}".format(kkt_norm), end="")
            PETSc.Sys.Print(" change: {:.6f}".format(change))

            change_arr.append(change)
            self.change = change
            self.f0val = f0val
            self.g0val = g0val
            self.loop = loop

            # print(f"rank: {rank} array {a_np}")
            # if np.all(np.array(change_arr[-10:]) < accepted_tol):
            #    break
            print(f"Time per iteration: {time.time() - t0}")

        with a_function.dat.vec as a_vec:
            a_vec.array_w = a_np
        # self.rf.set_local(new_params, a_np)
        PETSc.Sys.Print(
            "Optimization finished with change: {0:.5f} and iterations: {1}".format(
                change, loop
            )
        )
        return self.rf.controls.delist([a_function])

    def current_state(self):
        return (
            "It: {it}, obj: {obj:.3f} ".format(it=self.loop, obj=self.f0val)
            + "".join(
                [*(map("g[{0[0]}]: {0[1][0]:.3e} ".format, enumerate(self.g0val)))]
            )
            + " change: {:.3f}\n".format(self.change)
        )
