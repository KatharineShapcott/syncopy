# -*- coding: utf-8 -*-
# 
# Syncopy connectivity analysis methods
# 
# Created: 2019-07-24 14:22:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-24 17:29:22>

# Builtin/3rd party package imports
import sys
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser, unwrap_cfg
if __dask__:
    import dask.distributed as dd

# Module-wide output specs
complexDTypes = {"complex": np.complex128,
                 "abs": np.float32, 
                 "angle": np.float32, 
                 "imag": np.float32, 
                 "absimag": np.float32, 
                 "real": np.float32, 
                 "-logabs": np.float32}
complexConversions = {"complex": lambda x: x,
                      "abs": lambda x: np.abs(x, dtype=np.float32), 
                      "angle": lambda x: np.float32(np.angle(x)),
                      "imag": lambda x: np.float32(np.imag(x)),
                      "absimag": lambda x: np.abs(np.imag(x), dtype=np.float32),
                      "real": lambda x: np.float32(np.real(x)),
                      "-logabs": lambda x: -np.log(1 - np.abs(x)**2, dtype=np.float32)}

__all__ = ["connectivityanalysis"]


@unwrap_cfg
def connectivityanalysis(data, method='coh', partchannel=None, complex="abs"):
    """
    Coming soon...
    """
    
    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="SpectralData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")


def corr(trl_dat, timeAxis, pownorm=True,
         noCompute=False, chunkShape=None):
    """
    pownorm   = flag that specifies whether normalisation with the
                product of the power should be performed (thus should
                be true when correlation/coherence is requested, and
                false when covariance or cross-spectral density is
                requested).
    """
    
    pass