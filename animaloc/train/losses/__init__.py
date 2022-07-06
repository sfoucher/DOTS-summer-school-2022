from .register import LOSSES
from .ssim import *
from .focal import *
from .whd import *
from .distance import *
from .rmse import *

__all__ = ['LOSSES', *LOSSES.registry_names]