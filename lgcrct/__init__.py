"""
lgcrct — Local Geometric Consistency + Riemannian Centering Transformation.

Calibration-free cross-subject EEG transfer learning.

    from lgcrct import LGCRCTPipeline
    from lgcrct.smoothing import riemann_moving_average_blockwise
"""

from .pipeline import LGCRCTPipeline, infer_blocks_from_labels
from .smoothing import riemann_moving_average_blockwise
from .evaluation import run_loso

__version__ = "0.1.0"
__all__ = [
    "LGCRCTPipeline",
    "run_loso",
    "infer_blocks_from_labels",
    "riemann_moving_average_blockwise",
]
