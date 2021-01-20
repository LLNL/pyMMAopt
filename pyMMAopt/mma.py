"""
    mma.py - Copyright (C) 2016 University of Liege

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numexpr as ne
from scipy.sparse import spdiags
import time
from firedrake.petsc import PETSc
from mpi4py import MPI
from firedrake import COMM_SELF


print = lambda x: PETSc.Sys.Print(x, comm=COMM_SELF)


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
            "_timing": 1,
            "_elapsedTime": {
                "resKKT": -1,
                "preCompute": -1,
                "JacDual": -1,
                "RHSdual": -1,
                "nlIterPerEpsilon": [],
                "relaxPerNlIter": [],
                "timeEpsilonLoop": [],
                "mmasub": {"moveAsymp": -1, "moveLim": -1, "mmasubMat": -1, "all": -1},
                "subsolvIP": {"lin": -1, "relax": -1},
            },
        }

        # create the attributes
        for (prop, default) in param_defaults.items():
            setattr(self, prop, parameters.get(prop, default))
        self.local_n = len(self.xmin)
        self.xmin = np.array(self.xmin)
        self.xmax = np.array(self.xmax)
        self.comm = MPI.COMM_WORLD

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
        m,
        n,
        x,
        y,
        z,
        lam,
        xsi,
        eta,
        mu,
        zet,
        s,
        xmin,
        xmax,
        df0dx,
        fval,
        dfdx,
        a0,
        a,
        c,
        d,
    ):
        """
        The left hand sides of the KKT conditions for the following
        nonlinear programming problem are calculated.

        Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
        subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                    xmax_j <= x_j <= xmin_j,    j = 1,...,n
                    z >= 0,   y_i >= 0,         i = 1,...,m
        INPUT:
        m    : The number of general constraints.
        n    : The number of variables x_j.
        x    : List of current values of the n variables x_j.
        y    : List of current values of the m variables y_i.
        z    : List of current value of the single variable z.
        lam  : List of Lagrange multipliers for the m general constraints.
        xsi  : List of Lagrange multipliers for the n constraints xmin_j-x_j<=0
        eta  : List of Lagrange multipliers for the n constraints x_j-xmax_j<=0
        mu   : List of Lagrange multipliers for the m constraints -y_i<=0
        zet  : List of Lagrange multiplier for the single constraint -z<=0
        s    : List of Slack variables for the m general constraints.
        xmin : List of Lower bounds for the variables x_j.
        xmax : List of Upper bounds for the variables x_j.
        df0dx: List of of the derivatives of the objective function f_0
                with respect to the variables x_j, calculated at x.
        fval : List of the values of the constraint functions f_i,
                calculated at x.
        dfdx : (m x n)-List with the derivatives of the constraint functions
                f_i with respect to the variables x_j, calculated at x.
        dfdx(i,j) : List of the derivative of f_i with respect to x_j.
        a0   : Constant a_0 in the term a_0*z.
        a    : List of the constants a_i in the terms a_i*z.
        c    : List of the constants c_i in the terms c_i*y_i.
        d    : List of the constants d_i in the terms 0.5*d_i*(y_i)^2.

        OUTPUT:
        residual = (list) residual vector for the KKT conditions.
        """
        residual = []
        residual.extend(df0dx + np.dot(np.transpose(dfdx), lam) - xsi + eta)  # rex
        residual.extend(c + d * y - mu - lam)  # rey
        residual.append(self.a0 - zet - np.dot(a, lam))  # rez
        residual.extend(fval - self.a * z - y + s)  # relam
        residual.extend(xsi * (x - self.xmin))  # rexsi
        residual.extend(eta * (self.xmax - x))  # reeta
        residual.extend(mu * y)  # remu
        residual.append(zet * z)  # rezet
        residual.extend(lam * s)  # res
        return np.array(residual)

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
        x,
        y,
        z,
        lam,
        xsi,
        eta,
        mu,
        zet,
        s,
        epsi,
    ):
        Mdiag = self.Mdiag
        if self._timing:
            t0 = time.time()
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = 1.0 / ux1
        xlinv1 = 1.0 / xl1
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        local_gvec = P.dot(uxinv1) + Q.dot(xlinv1)
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
        local_residu_x = ne.evaluate("(dpsidx - Mdiag*xsi + Mdiag*eta)")
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
        residu_lam = gvec - self.a * z - y + s - b
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
        residu_zeta = zet * z - epsi
        residu_zeta_norm = residu_zeta ** 2
        residu_zeta_max = np.abs(residu_zeta)
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
            + residu_zeta_norm
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
                residu_zeta_max,
            )
        )

        if self._timing:
            self._elapsedTime["resKKT"] = time.time() - t0

        return residu_norm, residu_max

    def preCompute(
        self, alfa, beta, low, upp, p0, q0, P, Q, b, x, y, z, lam, xsi, eta, mu, s, epsi
    ):
        # delx,dely,delz,dellam,diagx,diagy,diagxinv,diaglamyi,GG):
        if self._timing:
            t0 = time.time()
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
        local_gvec = P.dot(uxinv1) + Q.dot(xlinv1)
        gvec = self.comm.allreduce(local_gvec, op=MPI.SUM)
        GG = ne.evaluate("uxinv2 * P - xlinv2 * Q")
        dpsidx = ne.evaluate("plam * uxinv2 - qlam * xlinv2")
        Mdiag = self.Mdiag
        delx = ne.evaluate(
            "dpsidx - Mdiag * epsi * invxalpha + Mdiag * epsi * invxbeta"
        )  # TODO mass matrix for xsi and eta
        diagx = ne.evaluate("plam / ux3 + qlam / xl3")
        diagx = ne.evaluate(
            "2 * diagx + Mdiag * xsi * invxalpha + Mdiag * eta * invxbeta"
        )  # TODO mass matrix for xsi and eta
        diagxinv = 1.0 / diagx

        dely = self.c + self.d * y - lam - epsi / y
        delz = self.a0 - np.dot(self.a, lam) - epsi / z
        dellam = gvec - self.a * z - y - b + epsi / lam
        diagy = self.d + mu / y
        diagyinv = 1.0 / diagy
        diaglam = s / lam
        diaglamyi = diaglam + diagyinv
        if self._timing:
            self._elapsedTime["preCompute"] = time.time() - t0

        return delx, dely, delz, dellam, diagx, diagy, diagxinv, diaglamyi, GG

    def JacDual(self, diagxinvGG, diaglamyi, GG, z, zet):
        """
        JAC = [Alam     a
                a'    -zet/z ]
        """
        if self._timing:
            t0 = time.time()
        local_Alam = np.dot(diagxinvGG, GG.T)
        Alam = self.comm.allreduce(local_Alam, op=MPI.SUM)
        mm = range(0, self.m)
        Alam[mm, mm] += diaglamyi
        jac = np.empty(shape=(self.m + 1, self.m + 1), dtype=float)
        jac[0 : self.m, 0 : self.m] = Alam
        jac[self.m, 0 : self.m] = self.a
        jac[self.m, self.m] = -zet / z
        jac[0 : self.m, self.m] = self.a
        if self._timing:
            self._elapsedTime["JacDual"] = time.time() - t0

        return jac

    def RHSdual(self, dellam, delx, dely, delz, diagxinvGG, diagy, GG):
        if self._timing:
            t0 = time.time()
        rhs = np.empty(shape=(self.m + 1,), dtype=float)
        local_diagxinvGG_delx = diagxinvGG.dot(delx)
        diagxinvGG_delx = self.comm.allreduce(local_diagxinvGG_delx, op=MPI.SUM)
        rhs[0 : self.m] = dellam + dely / diagy - diagxinvGG_delx
        rhs[self.m] = delz
        if self._timing:
            self._elapsedTime["RHSdual"] = time.time() - t0
        return rhs

    def getNewPoint(
        self,
        xold,
        yold,
        zold,
        lamold,
        xsiold,
        etaold,
        muold,
        zetold,
        sold,
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
        if self._timing:
            t0 = time.time()
        x = xold + step * dx
        y = yold + step * dy
        z = zold + step * dz
        lam = lamold + step * dlam
        xsi = xsiold + step * dxsi
        eta = etaold + step * deta
        mu = muold + step * dmu
        zet = zetold + step * dzet
        s = sold + step * ds
        if self._timing:
            self._elapsedTime["JacDual"] = time.time() - t0

        return x, y, z, lam, xsi, eta, mu, zet, s

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

        if self.IP > 0:
            print(str("*" * 80))
        if self._timing:
            self._elapsedTime["nlIterPerEpsilon"] = []
            self._elapsedTime["relaxPerNlIter"] = []
            self._elapsedTime["timeEpsilonLoop"] = []

        while epsi > self.epsimin:  # Loop over epsilon
            if self._timing:
                t0Eps = time.time()
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
                x,
                y,
                z,
                lam,
                xsi,
                eta,
                mu,
                zet,
                s,
                epsi,
            )

            # Solve the NL KKT problem for a given epsilon
            it_NL = 1
            relaxloopEpsi = []
            while residuNorm > 0.8 * epsi and it_NL < 200:
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
                    x,
                    y,
                    z,
                    lam,
                    xsi,
                    eta,
                    mu,
                    s,
                    epsi,
                )

                # assemble and solve the system: dlam or dx
                if self.m < self.local_n:
                    diagxinvGG = diagxinv * GG
                    AA = self.JacDual(diagxinvGG, diaglamyi, GG, z, zet)
                    bb = self.RHSdual(dellam, delx, dely, delz, diagxinvGG, diagy, GG)
                    if self._timing:
                        t0Solve = time.time()
                    solut = np.linalg.solve(AA, bb)
                    if self._timing:
                        self._elapsedTime["subsolvIP"]["lin"] = time.time() - t0Solve
                    dlam = solut[0 : self.m]
                    dz = solut[self.m]
                    # dx2 = - delx*diagxinv - np.transpose(GG).dot(dlam)/diagx
                    dx = -delx * diagxinv - np.dot((diagxinv * GG).T, dlam)
                else:
                    raise RuntimeError(
                        "This MMA implementation only handles a number of constraints smaller than the number of design variables"
                    )
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
                xold = np.copy(x)
                yold = np.copy(y)
                zold = np.copy(z)
                lamold = np.copy(lam)
                xsiold = np.copy(xsi)
                etaold = np.copy(eta)
                muold = np.copy(mu)
                zetold = np.copy(zet)
                sold = np.copy(s)

                # relaxation of the newton step for staying in feasible region
                len_xx = self.local_n * 2 + self.m * 4 + 2
                xx = np.zeros(len_xx)
                np.concatenate(
                    (y, [z], lam, xsi, eta, mu, [zet], s), out=xx
                )  # TODO probably it is not necessary to concatenate if you calculate the step separately
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
                    if self._timing:
                        t0_relax = time.time()
                    self.iPrint(
                        ["relax. it.", "Norm(res)", "step"],
                        [itto, resinewNorm, steg],
                        2,
                    )
                    # compute new point
                    x, y, z, lam, xsi, eta, mu, zet, s = self.getNewPoint(
                        xold,
                        yold,
                        zold,
                        lamold,
                        xsiold,
                        etaold,
                        muold,
                        zetold,
                        sold,
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
                        x,
                        y,
                        z,
                        lam,
                        xsi,
                        eta,
                        mu,
                        zet,
                        s,
                        epsi,
                    )

                    # update step
                    steg /= 2.0
                    itto += 1
                    if self._timing:
                        self._elapsedTime["subsolvIP"]["relax"] = time.time() - t0_relax

                if self._timing:
                    relaxloopEpsi.append(itto)
                self.iPrint(
                    ["relax. it.", "Norm(res)", "step"], [itto, resinewNorm, steg], 2
                )

                residuNorm = resinewNorm
                residuMax = resinewMax
                steg *= 2.0
                it_NL += 1
            if self._timing:
                self._elapsedTime["timeEpsilonLoop"].append(time.time() - t0Eps)
                self._elapsedTime["nlIterPerEpsilon"].append(it_NL - 1)
                self._elapsedTime["relaxPerNlIter"].append([relaxloopEpsi])

            if it_NL > 198:
                self.iPrint(["it limit", "with epsilon"], [it_NL, epsi], 0)
            epsi *= 0.1
            epsiIt += 1

        if self.IP > 0:
            print(str("*" * 80))

        return x, y, z, lam, xsi, eta, mu, zet, s

    def moveAsymp(self, xval, xold1, xold2, low, upp, iter):
        """
        Calculation of the asymptotes low and upp
        """
        if self._timing:
            t0 = time.time()
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
        if self._timing:
            self._elapsedTime["mmasub"]["moveAsymp"] = time.time() - t0

        return low, upp

    def moveLim(self, iter, xval, xold1, xold2, low, upp, factor):
        """
        Calculation of the move limits: alfa and beta
        """
        if self._timing:
            t0 = time.time()
        aa = np.maximum(
            low + self.albefa * (xval - low), xval - self.move * (self.xmax - self.xmin)
        )
        alfa = np.maximum(aa, self.xmin)
        aa = np.minimum(
            upp - self.albefa * (upp - xval), xval + self.move * (self.xmax - self.xmin)
        )
        beta = np.minimum(aa, self.xmax)
        if self._timing:
            self._elapsedTime["mmasub"]["moveLim"] = time.time() - t0

        return alfa, beta, factor

    def mmasubMat(self, xval, low, upp, df0dx, fval, dfdx):
        """
        Calculations of p0, q0, P, Q and b.
        """
        if self._timing:
            t0 = time.time()

        xmami = self.xmax - self.xmin
        # xmamieps = np.array([0.00001 * self.local_n])
        # xmami = np.maximum(xmami, xmamieps)
        xmamiinv = 1.0 / xmami
        ux1 = upp - xval
        ux2 = ux1 * ux1
        xl1 = xval - low
        xl2 = xl1 * xl1
        p0 = np.maximum(df0dx, 0)
        q0 = np.maximum(-df0dx, 0)
        pq0 = 0.001 * (p0 + q0) + self.raa0 * xmamiinv * self.Mdiag
        p0 = p0 + pq0
        q0 = q0 + pq0
        p0 = p0 * ux2
        q0 = q0 * xl2

        P = np.maximum(dfdx, 0)
        Q = np.maximum(-dfdx, 0)
        PQ = (
            0.001 * (P + Q)
            + self.Mdiag * self.raa0 * np.ones([self.m, 1]) * xmamiinv[np.newaxis, :]
        )
        P = ne.evaluate("ux2 * (P + PQ)")
        Q = ne.evaluate("xl2 * (Q + PQ)")
        ux1inv = ne.evaluate("1.0 / ux1")
        xl1inv = ne.evaluate("1.0 / xl1")
        local_b = np.dot(P, ux1inv) + np.dot(Q, xl1inv)
        b = self.comm.allreduce(local_b, op=MPI.SUM) - fval.T

        if self._timing:
            self._elapsedTime["mmasub"]["mmasubMat"] = time.time() - t0

        return p0, q0, P, Q, b

    def mmasub(
        self, xval, xold1, xold2, low, upp, f0val, fval, df0dx, dfdx, iter, factor
    ):
        """
        Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
        subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                    xmin_j <= x_j <= self.xmax_j, j = 1,...,n
                    z >= 0,   y_i >= 0, i = 1,...,m
        """
        if self._timing:
            t0 = time.time()

        # Calculation of the asymptotes low and upp
        low, upp = self.moveAsymp(xval, xold1, xold2, low, upp, iter)

        # Calculation of the bounds alfa and beta
        alfa, beta, factor = self.moveLim(iter, xval, xold1, xold2, low, upp, factor)

        # Calculations of p0, q0, P, Q and b
        p0, q0, P, Q, b = self.mmasubMat(xval, low, upp, df0dx, fval, dfdx)

        if self._timing:
            self._elapsedTime["mmasub"]["all"] = time.time() - t0

        return low, upp, alfa, beta, p0, q0, P, Q, b, factor

    def mma(
        self, xval, xold1, xold2, low, upp, f0val, fval, df0dx, dfdx, iter, factor=[]
    ):
        if self._timing:
            t0 = time.time()

        # generate subproblem
        low, upp, alfa, beta, p0, q0, P, Q, b, factor = self.mmasub(
            xval, xold1, xold2, low, upp, f0val, fval, df0dx, dfdx, iter, factor
        )

        # solve the subproblem
        x, y, z, lam, xsi, eta, mu, zet, s = self.subsolvIP(
            alfa, beta, low, upp, p0, q0, P, Q, b
        )

        if self._timing:
            self._elapsedTime["mma"] = time.time() - t0

        return x, y, z, lam, xsi, eta, mu, zet, s, low, upp, factor
