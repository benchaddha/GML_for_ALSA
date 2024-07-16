# ===----------------------------------------------------------------------===//
#
#                         Numbskull - Factor Graphs
#                       --------------------------------
#                               GML for ALSA
#
# init.py
#
# Stephen Bach
# Modified by: Benjamin Chaddha
#
# ===----------------------------------------------------------------------===//

"""inference and learning for factor graphs"""

from .numbskull import NumbSkull
from .numbskull import main
from .version import __version__

__all__ = ('numbskull', 'factorgraph', 'timer')
