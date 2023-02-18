
from typing import Optional, Literal, Callable
import torch
from scipy.interpolate import interp1d


INTERPOLATION = ["linear", "quadratic", "cubic"]
Interpolation = Literal["linear", "quadratic", "cubic"]
InterpolationCallable = Callable[[torch.Tensor], torch.Tensor]


class MREstimator(object):
    def effect(
        self,
        x: torch.Tensor,
        covars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the expected effect on the outcome when the exposure is set
        to X.

        E[Y | do(X=x)]

        In many cases, the ml-mr estimators condition on variables for
        estimation. If a sample of covariables is provided, the causal effect
        will be empirically averaged over the covariable values.

        i.e. Sum P(Y | C, do(X=x)) P(C) as empirically observed in the provided
        data.

        """
        raise NotImplementedError()

    @staticmethod
    def interpolate(
        xs: torch.Tensor,
        ys: torch.Tensor,
        mode: Interpolation = "cubic"
    ) -> InterpolationCallable:
        if mode not in INTERPOLATION:
            raise ValueError(f"Unknown interpolation type {mode}.")

        return interp1d(xs.numpy(), ys.numpy(), kind=mode, bounds_error=False)
