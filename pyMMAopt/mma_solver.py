# from fenics_adjoint import *
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import gather
from firedrake import PETSc

try:
    from .mma import MMAClient
except ImportError:
    print("You need to install MMA")
    raise
import numpy


class MMASolver(OptimizationSolver):
    def __init__(self, problem, parameters=None):
        OptimizationSolver.__init__(self, problem, parameters)

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
            "n": 1,
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
            "_timing": 1,
            "_elapsedTime": {
                "resKKT": -1,
                "preCompute": -1,
                "JacDual": -1,
                "JacPrim": -1,
                "RHSdual": -1,
                "nlIterPerEpsilon": [],
                "relaxPerNlIter": [],
                "timeEpsilonLoop": [],
                "mmasub": {"moveAsymp": -1, "moveLim": -1, "mmasubMat": -1, "all": -1},
                "subsolvIP": {"lin": -1, "relax": -1},
            },
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

        self.rfn = ReducedFunctionalNumPy(self.problem.reduced_functional)
        ncontrols = len(self.rfn.get_controls()) # TODO All gather called here, replaced with an AllReduce

        (self.lb, self.ub) = self.__get_bounds()
        (nconstraints, self.fun_g, self.jac_g) = self.__get_constraints()

        self.n = ncontrols
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

            for (bound, control) in zip(bounds, self.rfn.controls):
                general_lb, general_ub = bound  # could be float, Constant, or Function

                if isinstance(general_lb, (float, int)):
                    len_control = len(self.rfn.get_global(control)) # TODO All gather called here, replaced with an AllReduce, probably factor out the len_control
                    lb = numpy.array([float(general_lb)] * len_control)
                else:
                    lb = self.rfn.get_global(general_lb)# TODO All gather called here, replaced with an AllReduce

                lb_list.append(lb)

                if isinstance(general_ub, (float, int)):
                    len_control = len(self.rfn.get_global(control))# TODO All gather called here, replaced with an AllReduce, probably factor out the len_control
                    ub = numpy.array([float(general_ub)] * len_control)
                else:
                    ub = self.rfn.get_global(general_ub)

                ub_list.append(ub)

            ub = numpy.concatenate(ub_list)
            lb = numpy.concatenate(lb_list)

        else:
            # Unfortunately you really need to specify bounds, I think?!
            ncontrols = len(self.rfn.get_controls())
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
            ncontrols = len(self.rfn.get_controls())# TODO All gather called here, replaced with an AllReduce, probably factor out the len_control

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
        a_np = self.rfn.get_controls() # TODO get_controls involves an allgather of the vector.

        import numpy as np

        self.parameters["xmin"] = self.lb
        self.parameters["xmax"] = self.ub
        itermax = self.parameters["maximum_iterations"]

        # Create an optimizer client
        clientOpt = MMAClient(self.parameters)
        #'asyinit':0.2,'asyincr':0.8,'asydecr':0.3

        xold1 = np.copy(a_np)
        xold2 = np.copy(a_np)
        low = np.array([])
        upp = np.array([])

        change_arr = []

        while change > tol and loop <= itermax:
            f0val = self.rfn(a_np)
            df0dx = self.rfn.derivative(a_np)

            g0val = -1.0 * self.fun_g(a_np)
            j = self.jac_g(a_np)
            dg0dx = -1.0 * numpy.array(gather(j), dtype=float) # TODO, another gather that is not necessary (I believe)
            dg0dx = numpy.reshape(dg0dx, [self.m, self.n])

            # move limits
            clientOpt.xmin = np.maximum(self.lb, a_np - clientOpt.move) # TODO, All reduce on maximum
            clientOpt.xmax = np.minimum(self.ub, a_np + clientOpt.move) # TODO, All reduce on minimum

            xmma, y, z, lam, xsi, eta, mu, zet, s, low, upp, factor = clientOpt.mma(
                a_np, xold1, xold2, low, upp, f0val, g0val, df0dx, dg0dx, loop
            )

            change = np.abs(np.max(xmma - xold1))
            # update design variables
            xold2 = np.copy(xold1)
            xold1 = np.copy(a_np)
            a_np = np.copy(xmma)
            loop = loop + 1

            PETSc.Sys.Print("It: {it}, obj: {obj} ".format(it=loop, obj=f0val), end="")
            PETSc.Sys.Print(*(map("g[{0[0]}]: {0[1][0]} ".format, enumerate(g0val))), end="")
            PETSc.Sys.Print(" change: {:.3f}".format(change))

            change_arr.append(change)
            self.change = change
            self.f0val = f0val
            self.g0val = g0val
            self.loop = loop

            if np.all(np.array(change_arr[-10:]) < accepted_tol):
                break

        new_params = [control.copy_data() for control in self.rfn.controls]
        self.rfn.set_local(new_params, a_np)
        PETSc.Sys.Print(
            "Optimization finished with change: {0:.5f} and iterations: {1}".format(
                change, loop
            )
        )

        return self.rfn.controls.delist(new_params)

    def current_state(self):
        return (
            "It: {it}, obj: {obj:.3f} ".format(it=self.loop, obj=self.f0val)
            + "".join(
                [*(map("g[{0[0]}]: {0[1][0]:.3e} ".format, enumerate(self.g0val)))]
            )
            + " change: {:.3f}\n".format(self.change)
        )
