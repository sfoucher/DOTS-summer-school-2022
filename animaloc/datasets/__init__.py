from .register import DATASETS
from .csv import *
from .patched import *
from .folder import *
from .imagelevel import *

__all__ = ['DATASETS', *DATASETS.registry_names]