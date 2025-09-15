try:
    from importlib import metadata as _metadata
except ImportError:
    import importlib_metadata as _metadata 

try:
    __version__ = _metadata.version("argos3")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Main Components
from .convolutional import *
from .datagram import *
from .detector import *
from .encoder import *
from .formatter import *
from .lowpassfilter import *
from .matchedfilter import *    
from .modulator import *
from .multiplexer import *
from .preamble import *
from .sampler import * 
from .scrambler import *
from .synchronizer import *

# Channel
from .noise import *

# Transmitter
from .transmitter import *

# Receiver
from .receiver import *

# Extras
from .bersnr import *
from .data import * 
from .plotter import *
