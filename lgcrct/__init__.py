"""
lgcrct — Local Geometric Consistency + Riemannian Centering Transformation.

Unsupervised transductive cross-subject EEG transfer learning.

    from lgcrct import LGCRCTPipeline
    from lgcrct.lgc import local_riemannian_mean_blockwise
"""

from .pipeline import LGCRCTPipeline, MDMPipeline, infer_blocks_from_labels
from .lgc import local_riemannian_mean_blockwise
from .evaluation import run_loso

__version__ = "0.2.0"
__all__ = [
    "LGCRCTPipeline",
    "MDMPipeline",
    "run_loso",
    "infer_blocks_from_labels",
    "local_riemannian_mean_blockwise",
]
