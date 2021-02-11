from pyadjoint.optimization.constraints import InequalityConstraint
from firedrake import warning
from pyadjoint import stop_annotating


class ReducedInequality(InequalityConstraint):
    """This class represents constraints of the form
    Ghat(m) - Glimit >= 0
    where m is the parameter.
    For Ghat(m) - Glimit <= 0, pass lower=True
    """

    def __init__(self, Ghat, Glimit, Gcontrol, lower=True, normalized=True):
        self.Ghat = Ghat
        self.Glimit = float(Glimit)
        self.Gcontrol = Gcontrol
        self.lower = lower
        self.normalized = normalized
        if abs(Glimit) < 1e-4 and normalized:
            warning(f"Normalized is on with a very small bound {Glimit}")

    def function(self, m):

        # Compute the integral of the control over the domain
        integral = self.Gcontrol.tape_value()
        print(f"Value: {integral}, Constraint {self.Glimit}")
        with stop_annotating():
            if self.lower:
                if self.normalized:
                    value = -integral / self.Glimit + 1.0
                else:
                    value = -integral + self.Glimit
            else:
                if self.normalized:
                    value = integral / self.Glimit - 1.0
                else:
                    value = integral - self.Glimit
        return [value]

    def jacobian(self, m):

        with stop_annotating():
            gradients = self.Ghat.derivative()
            with gradients.dat.vec as v:
                if self.lower:
                    if self.normalized:
                        v.scale(-1.0 / self.Glimit)
                    else:
                        v.scale(-1.0)
                else:
                    if self.normalized:
                        v.scale(1.0 / self.Glimit)
                    else:
                        v.scale(1.0)
        return [gradients]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1
