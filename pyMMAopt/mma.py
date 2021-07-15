import numpy as np
import copy
import numexpr as ne
from firedrake.petsc import PETSc
from mpi4py import MPI
from firedrake import COMM_WORLD, warning


def print(x):
    return PETSc.Sys.Print(x, comm=COMM_WORLD)


class DesignState(object):
    state_args = [
        "x",
        "y",
        "z",
        "lam",
        "xsi",
        "eta",
        "mu",
        "zet",
        "s",
    ]

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            assert k in self.state_args, f"Variable {k} is not admitted"
            setattr(self, k, v)


class MMAClient(object):
    def __init__(self, parameters):
        """
        This package performs one MMA-iteration and solves the nonlinear
        programming problem written in the form:
              Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
            subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                        xmin_j <= x_j <= xmax_j,    j = 1,...,n
                        z >= 0,   y_i >= 0,         i = 1,...,m

        At a given iteration, the moving lower "low" and upper "upp"
        asymptotes are updated as follows:
            * the first two iterations:
                low_j = x_j - asyinit * (xmax - xmin)
                upp_j = x_j + asyinit * (xmax - xmin)
            * the later iterations:
                low_j = x_j - gamma_j * (xold_j - low_j)
                upp_j = x_j + gamma_j * (upp_j - xold_j)
              with
                zzz = (xval-xold1)*(xold1-xold2)
                gamma_j = asyincr if zzz>0; asydecr if zzz<0; 1 otherwise
              and finally
                low_j = maximum(low_j, x_j - 10*(xmax_j-xmin_j))
                low_j = minimum(low_j, x_j - 0.01*(xmax_j-xmin_j))
                upp_j = minimum(upp_j, x_j + 10*(xmax_j-xmin_j))
                upp_j = maximum(upp_j, x_j + 0.01*(xmax_j-xmin_j))

        All the parameters are provided in a dictionnary "parameter" s.t.
        "parameter = {
            m: "number of constraints",
            n: "number of variables x_j",
            xmin: "list with the lower bounds for the variables x_j",
            xmax: "list with the upper bounds for the variables x_j",
            a0: "constant in the term a_0*z",
            a: "list with the constants a_i in the terms a_i*z",
            c: "list with the constants c_i in the terms c_i*y_i",
            d: "list with the constants d_i in the terms 0.5*d_i*(y_i)^2",
            asyinit: "constant in the term update of low and upp",
            asyincr: "constant in the term update of low and upp",
            asydecr: "constant in the term update of low and upp",
            albefa: "constant in the term update of low and upp"
            move:"constant in the term update of low and upp"
        }
        """
        param_defaults = {
            "m": 1,
            "n": 1,
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
            "Mdiag": None,
            "gcmma": True,
        }

        # create the attributes
        for (prop, default) in param_defaults.items():
            setattr(self, prop, parameters.get(prop, default))
        self.local_n = len(self.xmin)
        if self.m > self.local_n:
            raise RuntimeError(
                "This MMA implementation only handles a number of constraints smaller than the number of design variables"
            )
        self.xmin = np.array(self.xmin)
        self.xmax = np.array(self.xmax)
        self.comm = MPI.COMM_WORLD

        # TODO if there are two variables per cell, the volume will be twice as big...
        # So far, this is not allowed by the element check in MMASolver
        local_volume = np.sum(self.Mdiag)
        self.volume = self.comm.allreduce(local_volume, op=MPI.SUM)
        print(f"Volume for MMA is: {self.volume}")

        # clasical configuration when parameters are unspecified
        if len(self.a) == 0:
            self.a = np.array([0.0] * self.m)
        if len(self.c) == 0:
            self.c = np.array([1000.0] * self.m)
        if len(self.d) == 0:
            self.d = np.array([1.0] * self.m)

    def iPrint(self, msgS, msg, level):
        if self.IP > level:
            print(
                str(" " * level)
                + " ".join(msgS[k] + ": {}".format(v) for k, v in enumerate(msg))
            )

    def residualKKTPrimal(
        self,
        x,
        y,
        z,
        lam,
        df0dx,
        fval,
        dfdx,
        residual_func,
        residual_plot
    ):
        
        residual_gradients_df0dx = (df0dx) / self.Mdiag
        print(f"df0dx contribution: {np.sum(residual_gradients_df0dx ** 2)}")

        residual_gradients_dfdx = np.dot(np.transpose(dfdx), lam) / self.Mdiag
        print(f"dfdx * lam contribution: {np.sum(residual_gradients_dfdx ** 2)}")
        with residual_func.dat.vec as res_vec:
            res_vec.array_w = residual_gradients_df0dx

        residual_plot.write(residual_func)

        residual_gradients = (df0dx + np.dot(np.transpose(dfdx), lam)) / self.Mdiag

        mu_min = np.where(residual_gradients > 0.0, residual_gradients, 0.0)
        mu_min *= (self.xmin - x) * np.sqrt(self.Mdiag)
        mu_max = np.where(residual_gradients < 0.0, -residual_gradients, 0.0)
        mu_max *= (self.xmax - x) * np.sqrt(self.Mdiag)
        norm2_grad = mu_min ** 2 + mu_max ** 2
        print(f"mu_min contribution: {np.sum(mu_min ** 2)}")
        print(f"mu_max contribution: {np.sum(mu_max ** 2)}")
        local_norm2 = np.sum(norm2_grad)
        print(f"local norm: {local_norm2}")
        norm2 = self.comm.allreduce(local_norm2, op=MPI.SUM)

        residual_constraints = fval - self.a * z - y
        residual_constraints = np.where(
            residual_constraints < 0.0, lam * residual_constraints, residual_constraints
        )
        return np.sqrt(np.sum(residual_constraints ** 2) + norm2)

    def resKKT(
        self,
        alfa,
        beta,
        low,
        upp,
        p0,
        q0,
        P,
        Q,
        b,
        design_state,
        epsi,
    ):
        x = design_state.x
        y = design_state.y
        z = design_state.z
        lam = design_state.lam
        xsi = design_state.xsi
        eta = design_state.eta
        mu = design_state.mu
        s = design_state.s
        zet = design_state.zet
        z = design_state.z

        Mdiag = self.Mdiag
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = 1.0 / ux1
        xlinv1 = 1.0 / xl1
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        local_gvec = (P * self.Mdiag).dot(uxinv1) + (Q * self.Mdiag).dot(xlinv1)
        gvec = self.comm.allreduce(local_gvec, op=MPI.SUM)
        dpsidx = ne.evaluate("plam / ux2 - qlam / xl2")

        def global_res_norm_square(local_residual):
            local_residuNorm = ne.evaluate("sum(local_residual ** 2)")
            residuNorm = self.comm.allreduce(local_residuNorm, op=MPI.SUM)
            return residuNorm

        def global_residual_max(local_residual):
            local_residuMax = np.linalg.norm(local_residual, np.inf)
            residuMax = self.comm.allreduce(local_residuMax, op=MPI.MAX)
            return residuMax

        # rex
        local_residu_x = ne.evaluate("(dpsidx * Mdiag - Mdiag*xsi + Mdiag*eta)")
        residu_x_norm = global_res_norm_square(
            ne.evaluate("local_residu_x / sqrt(Mdiag)")
        )  # This components is in the dual space, the norm has
        # to be weighted b the inverse of the mass matrix a.T * M^{-1} * a
        # do a sqrt if you're going to do **2 later
        residu_x_max = global_residual_max(local_residu_x)
        # rey
        residu_y = self.c + self.d * y - mu - lam
        residu_y_norm = np.sum(residu_y ** 2)
        residu_y_max = np.linalg.norm(residu_y, np.inf)
        # rez
        residu_z = self.a0 - zet - np.dot(self.a, lam)
        residu_z_norm = residu_z ** 2
        residu_z_max = np.abs(residu_z)
        # relam
        residu_lam = gvec - self.a * z - y + s + b
        residu_lam_norm = np.sum(residu_lam ** 2)
        residu_lam_max = np.linalg.norm(residu_lam, np.inf)
        # rexsi
        local_residu_xsi = ne.evaluate("(xsi * (x - alfa) - epsi) * sqrt(Mdiag)")
        residu_xsi_norm = global_res_norm_square(local_residu_xsi)
        residu_xsi_max = global_residual_max(local_residu_xsi)
        # reeta
        local_residu_eta = ne.evaluate("(eta * (beta - x) - epsi)* sqrt(Mdiag)")
        residu_eta_norm = global_res_norm_square(local_residu_eta)
        residu_eta_max = global_residual_max(local_residu_eta)
        # remu
        residu_mu = mu * y - epsi
        residu_mu_norm = np.sum(residu_mu ** 2)
        residu_mu_max = np.linalg.norm(residu_mu, np.inf)
        # rezet
        residu_zet = zet * z - epsi
        residu_zet_norm = residu_zet ** 2
        residu_zet_max = np.abs(residu_zet)
        # res
        residu_s = lam * s - epsi
        residu_s_norm = np.sum(residu_s ** 2)
        residu_s_max = np.linalg.norm(residu_s, np.inf)

        residu_norm = np.sqrt(
            residu_x_norm
            + residu_y_norm
            + residu_lam_norm
            + residu_xsi_norm
            + residu_eta_norm
            + residu_mu_norm
            + residu_s_norm
            + residu_z_norm
            + residu_zet_norm
        )
        residu_max = np.max(
            (
                residu_x_max,
                residu_y_max,
                residu_lam_max,
                residu_xsi_max,
                residu_eta_max,
                residu_mu_max,
                residu_s_max,
                residu_z_max,
                residu_zet_max,
            )
        )

        return residu_norm, residu_max

    def preCompute(self, alfa, beta, low, upp, p0, q0, P, Q, b, design_state, epsi):
        x = design_state.x
        y = design_state.y
        z = design_state.z
        lam = design_state.lam
        xsi = design_state.xsi
        eta = design_state.eta
        mu = design_state.mu
        s = design_state.s
        zet = design_state.zet
        z = design_state.z

        # delx,dely,delz,dellam,diagx,diagy,diagxinv,diaglamyi,GG):
        invxalpha = ne.evaluate("1 / (x - alfa)")
        invxbeta = ne.evaluate("1 / (beta - x)")
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        ux3 = ux1 * ux2
        xl3 = xl1 * xl2
        uxinv1 = 1.0 / ux1
        xlinv1 = 1.0 / xl1
        uxinv2 = 1.0 / ux2
        xlinv2 = 1.0 / xl2
        plam = p0 + lam.dot(P)
        qlam = q0 + lam.dot(Q)
        local_gvec = (P * self.Mdiag).dot(uxinv1) + (Q * self.Mdiag).dot(xlinv1)
        gvec = self.comm.allreduce(local_gvec, op=MPI.SUM)
        Mdiag = self.Mdiag
        GG = ne.evaluate("uxinv2 * P * Mdiag - xlinv2 * Q * Mdiag")
        dpsidx = ne.evaluate("plam * uxinv2 - qlam * xlinv2")
        delx = ne.evaluate(
            "dpsidx * Mdiag - Mdiag * epsi * invxalpha + Mdiag * epsi * invxbeta"
        )
        diagx = ne.evaluate(
            "2 * (plam / ux3 + qlam / xl3) * Mdiag + Mdiag * xsi * invxalpha + Mdiag * eta * invxbeta"
        )
        diagxinv = 1.0 / diagx

        dely = self.c + self.d * y - lam - epsi / y
        delz = self.a0 - np.dot(self.a, lam) - epsi / z
        dellam = gvec - self.a * z - y + b + epsi / lam
        diagy = self.d + mu / y
        diagyinv = 1.0 / diagy
        diaglam = s / lam
        diaglamyi = diaglam + diagyinv

        return delx, dely, delz, dellam, diagx, diagy, diagxinv, diaglamyi, GG

    def JacDual(self, diagxinvGG, diaglamyi, GG, z, zet):
        """
        JAC = [Alam     a
                a'    -zet/z ]
        """
        local_Alam = np.dot(diagxinvGG, GG.T)
        Alam = self.comm.allreduce(local_Alam, op=MPI.SUM)
        mm = range(0, self.m)
        Alam[mm, mm] += diaglamyi
        jac = np.empty(shape=(self.m + 1, self.m + 1), dtype=float)
        jac[0 : self.m, 0 : self.m] = Alam
        jac[self.m, 0 : self.m] = self.a
        jac[self.m, self.m] = -zet / z
        jac[0 : self.m, self.m] = self.a

        return jac

    def RHSdual(self, dellam, delx, dely, delz, diagxinvGG, diagy, GG):
        rhs = np.empty(shape=(self.m + 1,), dtype=float)
        local_diagxinvGG_delx = diagxinvGG.dot(delx)
        diagxinvGG_delx = self.comm.allreduce(local_diagxinvGG_delx, op=MPI.SUM)
        rhs[0 : self.m] = dellam + dely / diagy - diagxinvGG_delx
        rhs[self.m] = delz
        return rhs

    def getNewPoint(
        self,
        design_state_old,
        dx,
        dy,
        dz,
        dlam,
        dxsi,
        deta,
        dmu,
        dzet,
        ds,
        step,
    ):
        xold = design_state_old.x
        yold = design_state_old.y
        zold = design_state_old.z
        lamold = design_state_old.lam
        xsiold = design_state_old.xsi
        etaold = design_state_old.eta
        muold = design_state_old.mu
        sold = design_state_old.s
        zetold = design_state_old.zet
        zold = design_state_old.z

        x = xold + step * dx
        y = yold + step * dy
        z = zold + step * dz
        lam = lamold + step * dlam
        xsi = xsiold + step * dxsi
        eta = etaold + step * deta
        mu = muold + step * dmu
        zet = zetold + step * dzet
        s = sold + step * ds

        design_state = DesignState(
            x=x, y=y, z=z, lam=lam, xsi=xsi, eta=eta, mu=mu, zet=zet, s=s
        )

        return design_state

    def subsolvIP(self, alfa, beta, low, upp, p0, q0, P, Q, b):
        """
        This function subsolv solves the MMA subproblem with interior
        point method:

        minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
        + SUM[ ci*yi + 0.5*di*(yi)^2 ],

        subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
        alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.

        Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
        Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
        """
        # Initialize the variable values
        epsi = 1
        x = 0.5 * (alfa + beta)
        y = np.ones([self.m])
        z = 1
        lam = np.ones([self.m])
        xsi = 1.0 / (x - alfa)
        xsi = np.maximum(xsi, 1.0)
        eta = np.maximum(1.0 / (beta - x), 1.0)
        mu = np.maximum(np.ones([self.m]), 0.5 * self.c)
        zet = 1
        s = np.ones([self.m])
        epsiIt = 1

        design_state = DesignState(
            x=x, y=y, z=z, lam=lam, xsi=xsi, eta=eta, mu=mu, zet=zet, s=s
        )

        if self.IP > 0:
            print(str("*" * 80))

        while epsi > self.epsimin:  # Loop over epsilon
            self.iPrint(["Interior Point it.", "epsilon"], [epsiIt, epsi], 0)

            # compute residual
            residuNorm, residuMax = self.resKKT(
                alfa,
                beta,
                low,
                upp,
                p0,
                q0,
                P,
                Q,
                b,
                design_state,
                epsi,
            )

            # Solve the NL KKT problem for a given epsilon
            it_NL = 1
            relaxloopEpsi = []
            while residuNorm > 0.9 * epsi and it_NL < 200:
                self.iPrint(
                    ["NL it.", "Norm(res)", "Max(|res|)"],
                    [it_NL, residuNorm, residuMax],
                    1,
                )

                # precompute useful data -> time consuming!!!
                (
                    delx,
                    dely,
                    delz,
                    dellam,
                    diagx,
                    diagy,
                    diagxinv,
                    diaglamyi,
                    GG,
                ) = self.preCompute(
                    alfa,
                    beta,
                    low,
                    upp,
                    p0,
                    q0,
                    P,
                    Q,
                    b,
                    design_state,
                    epsi,
                )

                # assemble and solve the system: dlam or dx
                diagxinvGG = diagxinv * GG
                AA = self.JacDual(diagxinvGG, diaglamyi, GG, z, zet)
                bb = self.RHSdual(dellam, delx, dely, delz, diagxinvGG, diagy, GG)
                solut = np.linalg.solve(AA, bb)

                dlam = solut[0 : self.m]
                dz = solut[self.m]
                dx = -delx * diagxinv - np.dot((diagxinv * GG).T, dlam)
                dy = -dely / diagy + dlam / diagy
                dxsi = ne.evaluate(
                    "-xsi +  epsi / (x - alfa) - (xsi * dx) / (x - alfa)"
                )
                deta = ne.evaluate(
                    "-eta +  epsi / (beta - x) + (eta * dx) / (beta - x)"
                )
                dmu = -mu + epsi / y - (mu * dy) / y
                dzet = -zet + epsi / z - zet * dz / z
                ds = -s + epsi / lam - (s * dlam) / lam

                # store variables
                design_state_old = copy.copy(design_state)

                # relaxation of the newton step for staying in feasible region
                len_xx = self.local_n * 2 + self.m * 4 + 2
                xx = np.zeros(len_xx)
                np.concatenate((y, [z], lam, xsi, eta, mu, [zet], s), out=xx)
                dxx = np.zeros(len_xx)
                np.concatenate((dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds), out=dxx)

                stepxx = ne.evaluate("-1.01 * dxx / xx")
                local_stmxx = np.max(stepxx)
                stmxx = self.comm.allreduce(local_stmxx, op=MPI.MAX)
                stepalfa = ne.evaluate("-1.01 * dx / (x - alfa)")
                local_stmalfa = np.max(stepalfa)
                stmalfa = self.comm.allreduce(local_stmalfa, op=MPI.MAX)
                stepbeta = ne.evaluate("1.01 * dx / (beta - x)")
                local_stmbeta = np.max(stepbeta)
                stmbeta = self.comm.allreduce(local_stmbeta, op=MPI.MAX)
                stmalbe = np.maximum(stmalfa, stmbeta)
                stmalbexx = np.maximum(stmalbe, stmxx)
                stminv = np.maximum(stmalbexx, 1.0)
                steg = 1.0 / np.maximum(stmalbexx, 1.0)
                itto = 1
                resinewNorm = 2 * residuNorm
                resinewMax = 1e10
                while resinewNorm > residuNorm and itto < 200:
                    self.iPrint(
                        ["relax. it.", "Norm(res)", "step"],
                        [itto, resinewNorm, steg],
                        2,
                    )
                    # compute new point
                    design_state = self.getNewPoint(
                        design_state_old,
                        dx,
                        dy,
                        dz,
                        dlam,
                        dxsi,
                        deta,
                        dmu,
                        dzet,
                        ds,
                        steg,
                    )
                    x = design_state.x
                    y = design_state.y
                    z = design_state.z
                    lam = design_state.lam
                    xsi = design_state.xsi
                    eta = design_state.eta
                    mu = design_state.mu
                    s = design_state.s
                    zet = design_state.zet
                    z = design_state.z

                    # compute the residual
                    resinewNorm, resinewMax = self.resKKT(
                        alfa,
                        beta,
                        low,
                        upp,
                        p0,
                        q0,
                        P,
                        Q,
                        b,
                        design_state,
                        epsi,
                    )

                    # update step
                    steg /= 2.0
                    itto += 1

                    if itto > 198:
                        warning(f"Line search iteration limit {itto} reached")

                self.iPrint(
                    ["relax. it.", "Norm(res)", "step"], [itto, resinewNorm, steg], 2
                )

                residuNorm = resinewNorm
                residuMax = resinewMax
                steg *= 2.0
                it_NL += 1

            if it_NL > 198:
                warning(f"Iteration limit of the Newton solver ({it_NL}) reached")
            epsi *= 0.1
            epsiIt += 1

        if self.IP > 0:
            print(str("*" * 80))

        return x, y, z, lam

    def moveAsymp(self, xval, xold1, xold2, low, upp, iter):
        """
        Calculation of the asymptotes low and upp
        """
        if iter <= 2:
            low = xval - self.asyinit * (self.xmax - self.xmin)
            upp = xval + self.asyinit * (self.xmax - self.xmin)
        else:
            zzz = (xval - xold1) * (xold1 - xold2)
            factor = np.ones(self.local_n)
            factor[np.where(zzz > 0)] = self.asyincr
            factor[np.where(zzz < 0)] = self.asydecr
            low = xval - factor * (xold1 - low)
            upp = xval + factor * (upp - xold1)
            low = np.maximum(low, xval - 10 * (self.xmax - self.xmin))
            low = np.minimum(low, xval - 0.01 * (self.xmax - self.xmin))
            upp = np.minimum(upp, xval + 10 * (self.xmax - self.xmin))
            upp = np.maximum(upp, xval + 0.01 * (self.xmax - self.xmin))

        return low, upp

    def moveLim(self, iter, xval, xold1, xold2, low, upp, factor):
        """
        Calculation of the move limits: alfa and beta
        """
        aa = np.maximum(
            low + self.albefa * (xval - low), xval - self.move * (self.xmax - self.xmin)
        )
        alfa = np.maximum(aa, self.xmin)
        aa = np.minimum(
            upp - self.albefa * (upp - xval), xval + self.move * (self.xmax - self.xmin)
        )
        beta = np.minimum(aa, self.xmax)

        return alfa, beta, factor

    def mmasubMat(self, xval, low, upp, f0val, df0dx, fval, dfdx, rho0, rhoi):
        """
        Calculations of p0, q0, P, Q and b.
        """

        xmami = self.xmax - self.xmin
        xmamiinv = 1.0 / xmami
        ux1 = upp - xval
        ux2 = ux1 * ux1
        xl1 = xval - low
        xl2 = xl1 * xl1
        p0 = np.maximum(df0dx, 0.0)
        q0 = np.maximum(-df0dx, 0.0)
        pq0 = 0.001 * (p0 + q0) + rho0 * xmamiinv
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2

        P = np.maximum(dfdx, 0.0)
        Q = np.maximum(-dfdx, 0.0)
        PQ = 0.001 * (P + Q) + rhoi[:, np.newaxis] * xmamiinv[np.newaxis, :]
        P = ne.evaluate("ux2 * (P + PQ)")
        Q = ne.evaluate("xl2 * (Q + PQ)")
        ux1inv = ne.evaluate("1.0 / ux1")
        xl1inv = ne.evaluate("1.0 / xl1")

        local_b0 = np.dot(p0 * self.Mdiag, ux1inv) + np.dot(q0 * self.Mdiag, xl1inv)
        b0 = -self.comm.allreduce(local_b0, op=MPI.SUM) + f0val

        local_b = np.dot(P * self.Mdiag, ux1inv) + np.dot(Q * self.Mdiag, xl1inv)
        b = -self.comm.allreduce(local_b, op=MPI.SUM) + fval.T

        return p0, q0, P, Q, b0, b

    def calculate_initial_rho(self, dfdx, xmax, xmin):
        local_rho = np.dot(np.abs(dfdx), xmax - xmin)
        rho = 0.1 / self.volume * self.comm.allreduce(local_rho, op=MPI.SUM)
        if self.gcmma == False:
            if isinstance(rho, np.ndarray):
                rho.fill(1e-5)
            else:
                rho = 1e-5
        return rho

    def calculate_rho(self, rho, new_fval, fapp, x_inner, x_outer, low, upp):
        denom = np.dot(
            self.Mdiag,
            (
                (upp - low)
                * (x_inner - x_outer) ** 2
                / ((upp - x_inner) * (x_inner - low) * (self.xmax - self.xmin))
            ),
        )
        denom = self.comm.allreduce(denom, op=MPI.SUM)
        delta = (new_fval - fapp) / denom

        if not isinstance(fapp, np.ndarray):
            delta = np.array([delta])
            rho

        return np.where(
            delta > 0,
            np.minimum(1.1 * (rho + delta), 10.0 * rho),
            rho,
        )

    def convex_approximation(self, x_inner, p, q, b, low, upp):
        if len(p.shape) > 1:
            local_fapp = np.sum(
                self.Mdiag * p / (upp - x_inner) + self.Mdiag * q / (x_inner - low), 1
            )
        else:
            local_fapp = np.sum(
                self.Mdiag * p / (upp - x_inner) + self.Mdiag * q / (x_inner - low)
            )
        fapp = self.comm.allreduce(local_fapp, op=MPI.SUM) + b
        return fapp

    def condition_check(self, fapp, new_fval):

        if isinstance(fapp, np.ndarray):
            assert fapp.size == new_fval.size
        else:
            fapp = np.array([fapp])
            new_fval = np.array([new_fval])

        tolerance = 1e-8

        condition = False
        for fapp_i, new_fval_i in zip(fapp, new_fval):
            print(f"condition: fapp {fapp_i}, new_fval {new_fval_i}")
            if fapp_i + tolerance >= new_fval_i:
                condition = True
            else:
                return False

        return condition

    def mma(
        self,
        xval,
        xold1,
        xold2,
        low,
        upp,
        f0val,
        fval,
        df0dx,
        dfdx,
        iter,
        factor=[],
        eval_f=None,
        eval_g=None,
    ):

        # Calculation of the asymptotes low and upp
        low, upp = self.moveAsymp(xval, xold1, xold2, low, upp, iter)

        # Calculation of the bounds alfa and beta
        alfa, beta, factor = self.moveLim(iter, xval, xold1, xold2, low, upp, factor)

        rho0 = self.calculate_initial_rho(df0dx, self.xmax, self.xmin)
        rhoi = self.calculate_initial_rho(dfdx, self.xmax, self.xmin)

        inner_it_max = 100
        inner_it = 0
        # Apply Riesz map to the gradients
        df0dx /= self.Mdiag
        dfdx /= self.Mdiag
        while inner_it < inner_it_max:
            # generate subproblem
            p0, q0, P, Q, b0, b = self.mmasubMat(
                xval, low, upp, f0val, df0dx, fval, dfdx, rho0, rhoi
            )
            print(f"rho0: {rho0}, rhoi: {rhoi}")

            # solve the subproblem
            x_inner, y, z, lam = self.subsolvIP(alfa, beta, low, upp, p0, q0, P, Q, b)

            new_f0val = eval_f(x_inner)
            new_fval = eval_g(x_inner).flatten()

            f0app = self.convex_approximation(x_inner, p0, q0, b0, low, upp)
            fapp = self.convex_approximation(x_inner, P, Q, b, low, upp)

            assert fapp.size == new_fval.size
            if self.gcmma == False or (
                self.condition_check(f0app, new_f0val)
                and self.condition_check(fapp, new_fval)
            ):
                break
            else:
                rho0 = self.calculate_rho(
                    rho0, new_f0val, f0app, x_inner, xval, low, upp
                )
                rhoi = self.calculate_rho(rhoi, new_fval, fapp, x_inner, xval, low, upp)
                print(f"Recalculating rho")

            inner_it += 1

        return (
            x_inner,
            y,
            z,
            lam,
            low,
            upp,
            factor,
            new_f0val,
            new_fval,
        )
