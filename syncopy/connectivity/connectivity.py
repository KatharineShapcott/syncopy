# -*- coding: utf-8 -*-
# 
# Syncopy connectivity analysis methods
# 
# Created: 2019-07-24 14:22:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-25 17:32:22>

# Builtin/3rd party package imports
import sys
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser, unwrap_cfg
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import SpectralData
from syncopy import __dask__
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

    # Ensure a valid computational method was selected
    avail_methods = ["coh", "wavelet"]
    if method not in avail_methods:
        lgl = "'" + "or '".join(opt + "' " for opt in avail_methods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Get positional indices of dimensions in `data` relative to class defaults
    dimord = [SpectralData().dimord.index(dim) for dim in data.dimord]

def corr(trl_dat, dimord, pownorm=True, complex="abs",
         noCompute=False, chunkShape=None, inMemory=True):
    """
    pownorm   = flag that specifies whether normalisation with the
                product of the power should be performed (thus should
                be true when correlation/coherence is requested, and
                false when covariance or cross-spectral density is
                requested).
    """
    
    if dimord != list(range(len(dimord))):
        shp = tuple([trl_dat.shape[dim] for dim in dimord])
        reorder = True
    else:
        shp = trl_dat.shape
        reorder = False
    
    (_, nTaper, nFreq, nChannels) = shp
    outShape = (1, 1, nChannels, nChannels, nFreq)
    
    if noCompute:
        return outShape, complexDTypes[complex]
    
    if inMemory:
    
        if reorder:
            dat = np.moveaxis(trl_dat, dimord, list(range(len(dimord))))
        else:
            dat = trl_dat
        
        conn = np.full(chunkShape, np.nan, dtype=complexDTypes[complex])
        for nf in range(nFreq):
            tmp = np.dot(np.squeeze(dat[:, :, nf, :]), np.squeeze(dat[:, :, nf, :]).T)/nTaper
            if pownorm:
                tdg = np.diag(tmp)        
                tmp /= np.sqrt(np.repeat(tdg.rehape(-1, 1), axis=1, repeats=nChannels) *\
                            np.repeat(tdg.rehape(1, -1), axis=0, repeats=nChannels))
            conn[:, :, :, :, nf] = complexConversions[complex](tmp)
        
        return conn
    
    else:
        pass
    


class ConnectivityCorr(ComputationalRoutine):
    
    computeFunction = staticmethod(corr)
    
    def process_metadata(self, data, out):

        # Some index gymnastics to get trial begin/end "samples"
        if self.keeptrials:
            time = np.arange(len(data.trials))
            time = time.reshape((time.size, 1))
            out.sampleinfo = np.hstack([time, time + 1])
            out.trialinfo = np.array(data.trialinfo)
            out._t0 = np.zeros((len(data.trials),))
        else:
            out.sampleinfo = np.array([[0, 1]])
            out.trialinfo = out.sampleinfo[:, 3:]
            out._t0 = np.array([0])

        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel)
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[out.dimord.index("taper")])
        if self.cfg["foi"] is not None:
            out.freq = self.cfg["foi"]
        else:
            nFreqs = self.outputShape[out.dimord.index("freq")]
            out.freq = np.linspace(0, 1, nFreqs) * (data.samplerate / 2)
            
