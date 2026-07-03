"""models_v2 registry.

  B0_baserate  base rate / default rule
  B1_static    static (x, g, theta)
  B2_mean      B1 + mean-pooled probe history (weak baseline)
  B2_deepsets  torch DeepSets set encoder over probe history (permutation invariant)
  B2_gru       torch GRU over probe history (order sensitive)
  OracleZ      static + true damping (upper bound; audit only)
"""

from .numpy_models import B0, B1, B2Mean, OracleZ
from .torch_models import GRU, DeepSets

REGISTRY = {
    B0.name: B0,
    B1.name: B1,
    B2Mean.name: B2Mean,
    DeepSets.name: DeepSets,
    GRU.name: GRU,
    OracleZ.name: OracleZ,
}

# models legal to compare as "the method" (exclude OracleZ which reads the secret)
DEPLOYABLE = [B0.name, B1.name, B2Mean.name, DeepSets.name, GRU.name]

__all__ = ["B0", "B1", "B2Mean", "DeepSets", "GRU", "OracleZ", "REGISTRY", "DEPLOYABLE"]
