from pyadjoint.adjfloat import AdjFloat
from pyadjoint.optimization.optimization_solver import OptimizationSolver
from firedrake.petsc import PETSc
import firedrake as fd
from pyadjoint import stop_annotating
from firedrake import COMM_WORLD, HDF5File
from mpi4py import MPI
import time
import signal


try:
    from .mma import MMAClient
except ImportError:
    print("You need to install MMA")
    raise
import numpy


def print(x):
    return PETSc.Sys.Print(x, comm=COMM_WORLD)


def copy_vec_into_funct(func, vec):
    with func.dat.vec as a_vec:
        a_vec.array_w = vec


def func_to_vec(func):
    with func.dat.vec_ro as func_vec:
        vec = func_vec.array
    return vec


class MMASolver(OptimizationSolver):
    def __init__(self, problem, parameters=None):
        OptimizationSolver.__init__(self, problem, parameters)

        self.rf = self.problem.reduced_functional
        if len(self.rf.controls) > 1:
            raise RuntimeError("Only one control is possible for MMA")

        if isinstance(self.rf.controls[0].control, fd.Function) is False:
            raise RuntimeError("Only control of type Function is possible for MMA")

        control_funcspace = self.rf.controls[0].control.function_space()
        control_elem = control_funcspace.ufl_element()

        supported_fe = ["DQ", "Discontinuous Lagrange"]
        mass_matrix_support = True
        if control_elem.family() == "TensorProductElement":
            sub_elem = control_elem.sub_elements()
            if (
                sub_elem[0].family() not in supported_fe
                or sub_elem[0].degree() != 0
                or sub_elem[1].family() not in supported_fe
                or sub_elem[1].degree() != 0
            ):
                mass_matrix_support = False
        elif control_elem.family() not in supported_fe or control_elem.degree() != 0:
            mass_matrix_support = False

        if parameters.get("norm") == "L2":
            if mass_matrix_support:
                with stop_annotating():
                    self.Mdiag = fd.assemble(
                        fd.TrialFunction(control_funcspace)
                        * fd.TestFunction(control_funcspace)
                        * fd.dx,
                        diagonal=True,
                    ).dat.data_ro
            else:
                fd.warning(
                    "Only zero degree Discontinuous Galerkin function space is supported for norm = L2"
                )
                self.Mdiag = numpy.ones(self.rf.controls[0].control.dat.data_ro.size)
        else:
            self.Mdiag = numpy.ones(self.rf.controls[0].control.dat.data_ro.size)

        self.__build_mma_problem()
        self.__set_parameters()

        self.change = 0.0
        self.f0val = 0.0
        self.g0val = [[0.0] for _ in range(parameters["m"])]
        self.loop = 0

    def __set_parameters(self):
        """Set some basic parameters from the parameters dictionary that the user
        passed in, if any."""
        param_defaults = {
            "m": 1,
            "n": 1,
            "Mdiag": False,
            "tol": 1e-8,
            "rfunctol": 1e-8,
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
            "norm": "L2",
            "gcmma": False,
            "output_dir": "./",
            "restart_file": False,
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
            self.rf.controls[0].control, fd.Function
        ), "Only control of type Function is possible for MMA"

        (self.lb, self.ub) = self.__get_bounds()
        (nconstraints, self.fun_g, self.jac_g) = self.__get_constraints()

    def __get_bounds(self):
        r"""Convert the bounds into the format accepted by MMA (two numpy arrays,
        one for the lower bound and one for the upper)."""

        bounds = self.problem.bounds

        if bounds is not None:
            lb_list = []
            ub_list = []  # a list of numpy arrays, one for each control

            for (bound, control) in zip(bounds, self.rf.controls):
                general_lb, general_ub = bound  # could be float, Constant, or Function

                if isinstance(control.control, fd.Function):
                    n_local_control = control.control.dat.data_ro.size
                elif isinstance(control.control, (fd.Constant, AdjFloat)):
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
                if not flag:
                    return empty

                rows = numpy.array([], dtype=int)
                cols = numpy.array([], dtype=int)
                return (rows, cols)

            return (nconstraints, fun_g, jac_g)
        else:
            # The length of the constraint vector
            nconstraints = constraint._get_constraint_dim()
            # The constraint function

            def fun_g(x, user_data=None):
                return numpy.array(constraint.function(x), dtype=float)

            # The constraint Jacobian:
            # flag = True  means 'tell me the sparsity pattern';
            # flag = False means 'give me the damn Jacobian'.

            def jac_g(x, user_data=None):
                return constraint.jacobian(x)

            return (nconstraints, fun_g, jac_g)

    def solve(
        self, xold1_func=None, xold2_func=None, low_func=None, upp_func=None, loop=0
    ):

        assert (
            xold1_func is None
            and xold2_func is None
            and low_func is None
            and upp_func is None
        ) or (xold1_func and xold2_func and low_func and upp_func)

        parameters = self.parameters
        tol = parameters["tol"]
        rfunctol = parameters["rfunctol"]
        # Initial estimation
        control_function = self.rf.controls[0].control

        if not xold1_func:
            xold1_func = fd.Function(control_function.function_space())
            xold2_func = fd.Function(control_function.function_space())
            low_func = fd.Function(control_function.function_space())
            upp_func = fd.Function(control_function.function_space())

        if parameters.get("restart_file", None):
            with HDF5File(parameters["restart_file"], "r") as checkpoint:
                checkpoint.read(control_function, "/control")
                checkpoint.read(xold1_func, "/xold1_func")
                checkpoint.read(xold2_func, "/xold2_func")
                checkpoint.read(low_func, "/low_func")
                checkpoint.read(upp_func, "/upp_func")

        with control_function.dat.vec_ro as control_vec:
            a_np = control_vec.array

        xold1 = func_to_vec(xold1_func)
        xold2 = func_to_vec(xold2_func)
        low = func_to_vec(low_func)
        upp = func_to_vec(upp_func)

        import numpy as np

        parameters["xmin"] = self.lb
        parameters["xmax"] = self.ub
        parameters["n"] = control_function.dat.data_ro.size
        parameters["Mdiag"] = self.Mdiag
        itermax = parameters["maximum_iterations"]

        # Create an optimizer client
        clientOpt = MMAClient(parameters)
        # 'asyinit':0.2,'asyincr':0.8,'asydecr':0.3

        change_arr = []

        a_function = control_function.copy(deepcopy=True)

        def receive_signal(signum, stack):
            copy_vec_into_funct(xold1_func, xold1)
            copy_vec_into_funct(xold2_func, xold2)
            copy_vec_into_funct(low_func, low)
            copy_vec_into_funct(upp_func, upp)

            with HDF5File(
                f"{parameters['output_dir']}/checkpoint_iter_{loop}.h5", "w"
            ) as checkpoint:
                checkpoint.write(a_function, "/control")
                checkpoint.write(xold1_func, "/xold1_func")
                checkpoint.write(xold2_func, "/xold2_func")
                checkpoint.write(low_func, "/low_func")
                checkpoint.write(upp_func, "/upp_func")

        signal.signal(signal.SIGUSR1, receive_signal)

        def eval_f(a_np):
            with a_function.dat.vec as a_vec:
                a_vec.array_w = a_np
            return self.rf(a_function)

        def eval_g(a_np):
            with a_function.dat.vec as a_vec:
                a_vec.array_w = a_np
            return -1.0 * self.fun_g(a_function)

        n = parameters["n"]
        m = parameters["m"]
        dg0dx = np.empty([m, n])
        df0dx = np.empty([n])
        comm = MPI.COMM_WORLD

        f0val = eval_f(a_np)
        g0val = eval_g(a_np).flatten()

        change = 1.0
        rfunc_change = 1.0
        prev_f0val = f0val
        while change > tol and loop <= itermax and rfunc_change > rfunctol:
            t0 = time.time()

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

            (xmma, y, z, lam, low, upp, factor, f0val, g0val,) = clientOpt.mma(
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
            rfunc_change = abs(prev_f0val - f0val) / abs(prev_f0val)
            prev_f0val = f0val
            # update design variables
            xold2 = np.copy(xold1)
            xold1 = np.copy(a_np)
            a_np = np.copy(xmma)
            loop = loop + 1

            PETSc.Sys.Print("It: {it}, obj: {obj} ".format(it=loop, obj=f0val), end="")
            PETSc.Sys.Print(
                "".join([f"g[{index}]: {value} " for index, value in enumerate(g0val)])
            )
            # PETSc.Sys.Print(" Inner iterations: {:d}".format(inner_it), end="")
            PETSc.Sys.Print(" kkt: {:6f}".format(kkt_norm), end="")
            PETSc.Sys.Print(" change: {:.6f}".format(change), end="")
            PETSc.Sys.Print(" rel obj change: {:.6f}".format(rfunc_change))

            change_arr.append(change)
            self.change = change
            self.f0val = f0val
            self.g0val = g0val
            self.loop = loop

            # print(f"rank: {rank} array {a_np}")
            # if np.all(np.array(change_arr[-10:]) < accepted_tol):
            #    break
            print(f"Time per iteration: {time.time() - t0}")

        copy_vec_into_funct(a_function, a_np)
        copy_vec_into_funct(xold1_func, xold1)
        copy_vec_into_funct(xold2_func, xold2)
        copy_vec_into_funct(low_func, low)
        copy_vec_into_funct(upp_func, upp)
        # self.rf.set_local(new_params, a_np)
        PETSc.Sys.Print(
            "Optimization finished with change: {0:.5f} and iterations: {1}".format(
                change, loop
            )
        )
        results = {
            "control": self.rf.controls.delist([a_function]),
            "xold1": xold1_func,
            "xold2": xold2_func,
            "low": low_func,
            "upp": upp_func,
            "loop": loop,
        }
        return results

    def current_state(self):
        return (
            "It: {it}, obj: {obj:.3f} ".format(it=self.loop, obj=self.f0val)
            + "".join(
                [*(map("g[{0[0]}]: {0[1][0]:.3e} ".format, enumerate(self.g0val)))]
            )
            + " change: {:.3f}\n".format(self.change)
        )
