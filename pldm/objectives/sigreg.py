"""
pldm/objectives/sigreg.py — SIGReg objective for HWM_PLDM.

Spectral Isotropic Gaussian Regularisation (from the LeWM paper).
Enforces that encoder output z ~ N(0, I) via random projections, which
structures the latent space so that cosine distance is geometrically
meaningful for CEM/MPPI planning.

Algorithm:
  1. Sample M unit random vectors W ∈ R^{D×M}
  2. Project: p = z @ W  →  (B, M)
  3. mean_loss = ||mean(p, dim=0)||²  (push projections to zero mean)
  4. std_loss  = ||std(p, dim=0) - 1||²  (push projections to unit std)
  5. total = global_coeff * (mean_loss + std_loss)

Applied to context encodings only (first timestep in the sequence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple

import torch
import torch.nn.functional as F

from pldm.configs import ConfigBase
from pldm.models.jepa import ForwardResult
from pldm.models.utils import flatten_conv_output


class SIGRegLossInfo(NamedTuple):
    total_loss: torch.Tensor
    mean_loss: torch.Tensor
    std_loss: torch.Tensor
    loss_name: str = "sigreg"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_mean_loss": self.mean_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_std_loss": self.std_loss.item(),
        }


@dataclass
class SIGRegObjectiveConfig(ConfigBase):
    n_proj: int = 1024       # M — number of random projections
    global_coeff: float = 0.1  # λ — loss weight (LeWM paper value)


class SIGRegObjective(torch.nn.Module):
    """
    SIGReg regularisation on the encoder's context-frame output.

    Encourages z ~ N(0, I) so cosine distance in latent space is
    a reliable proxy for state dissimilarity during planning.
    """

    def __init__(
        self,
        config: SIGRegObjectiveConfig,
        repr_dim: int,
        name_prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        # repr_dim may be a tuple for conv outputs; resolve to scalar
        if isinstance(repr_dim, (tuple, list)):
            import operator, functools
            repr_dim = functools.reduce(operator.mul, repr_dim)
        self.repr_dim = repr_dim

    def __call__(self, _batch, result: List[ForwardResult]) -> SIGRegLossInfo:
        # Use the highest-level result's context encodings
        encodings = result[-1].backbone_output.encodings  # (T, B, ...) or (B, ...)

        flat = flatten_conv_output(encodings)  # (T, B, D) or (B, D)

        # Take context frame only (index 0 along T if present)
        if flat.dim() == 3:
            z = flat[0]   # (B, D)
        else:
            z = flat       # (B, D)

        mean_loss, std_loss = self._sigreg(z)
        total = self.config.global_coeff * (mean_loss + std_loss)

        return SIGRegLossInfo(
            total_loss=total,
            mean_loss=mean_loss,
            std_loss=std_loss,
            name_prefix=self.name_prefix,
        )

    def _sigreg(self, z: torch.Tensor):
        """z: (B, D) — returns (mean_loss, std_loss) scalars."""
        B, D = z.shape
        W = torch.randn(D, self.config.n_proj, device=z.device, dtype=z.dtype)
        W = F.normalize(W, dim=0)          # unit columns
        proj = z @ W                        # (B, M)
        mean_loss = proj.mean(0).pow(2).mean()
        std_loss  = (proj.std(0) - 1.0).pow(2).mean()
        return mean_loss, std_loss
