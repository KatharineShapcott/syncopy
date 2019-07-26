# -*- coding: utf-8 -*-
# 
# Syncopy connectivity analysis methods
# 
# Created: 2019-07-24 14:22:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-26 17:14:53>

# Builtin/3rd party package imports
import os
import sys
import h5py
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser, unwrap_cfg
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import SpectralData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy import __dask__
if __dask__:
    import dask.distributed as dd
    import dask.array as da

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
    avail_methods = ["coh", "corr", "cov", "csd"]
    if method not in avail_methods:
        lgl = "'" + "or '".join(opt + "' " for opt in avail_methods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)
    
    # Set power-normalization based on chosen method
    if method in ["cov", "csd"]:
        pownorm = False
    else:
        pownorm = True
    
    # Ensure output selection for complex-valued data makes sense
    options = complexConversions.keys()
    if complex not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="complex", actual=complex)
        

    # Get positional indices of dimensions in `data` relative to class defaults
    dimord = [data.dimord.index(dim) for dim in SpectralData().dimord]
    
    if method in ["coh", "corr", "cov", "csd"]:
        # check input
        pass
    
    # Construct dict of classes of available methods
    if method in ["coh", "corr", "cov", "csd"]:
        connMethod = ConnectivityCorr(dimord, **mth_input)

    # Detect if dask client is running to set `parallel` keyword below accordingly
    try:
        dd.get_client()
        use_dask = True
    except ValueError:
        use_dask = False

    # Perform actual computation
    connMethod.initialize(data)
    connMethod.compute(data, out, parallel=use_dask, log_dict=log_dct)
    

def corr(trl_dat, dimord, pownorm=True, complex="abs",
         noCompute=False, chunkShape=None, inMemory=True):
    """
    pownorm   = flag that specifies whether normalisation with the
                product of the power should be performed (thus should
                be true when correlation/coherence is requested, and
                false when covariance or cross-spectral density is
                requested).
    """
    
    # Determine dimensional order of input and (if necessary) re-arrange shape-tuple
    # of `trl_dat` to be able to identify taper-, frequency-, and channel-counts
    if dimord != list(range(len(dimord))):
        shp = tuple([trl_dat.shape[dim] for dim in dimord])
    else:
        shp = trl_dat.shape
    
    # Set expected shape of output and get outta here if we're in the dry-run phase 
    (_, nTaper, nFreq, nChannel) = shp
    outShape = (1, nFreq, nChannel, nChannel)
    outdtype = complexDTypes[complex]
    if noCompute:
        return outShape, outdtype

    # See if we can parallelize across frequencies    
    try:
        dd.get_client()
        parallel = True
    except ValueError:
        parallel = False
        
    # Get index of frequency dimension in input
    freqidx = dimord[SpectralData().dimord.index("freq")] 
        
    if parallel:
        
        # Create a dask array chunked by frequency and map each taper-channel
        # block onto `_corr` to compute channel x channel coherence/covariance
        chunks = list(trl_dat.shape)
        chunks[freqidx] = 1
        dat = da.from_array(trl_dat, chunks=tuple(chunks))
        
        conn = dat.map_blocks(_corr, nChannel, nTaper, pownorm, 
                              dtype=outdtype, 
                              chunks=(1, 1, nChannel, nChannel))
        conn = conn.reshape(1, nFreq, nChannel, nChannel)
        
        # Stacking gymnastics in case trials have different lengths (i.e., frequency-counts)
        if nFreq < outShape[1]:
            conn = da.hstack([conn, 
                              da.zeros((1, outShape[1] - nFreq, nChannel, nChannel), 
                                       dtype=conn.dtype)])
            
    else:

        # If trials don't fit into memory, `trl_dat` is already an HDF5 dataset -
        # create a new HDF5 file parallel to `trl_dat`'s parent file and write results
        # directly to it - flush after each frequency pass to not overflow memory
        idx = [slice(None)] * len(dimord)
        if inMemory:
            conn = np.full(chunkShape, np.nan, dtype=outdtype)
        else:
            h5f = h5py.File(trl_dat.file.filename.replace(".h5", "_result.h5"))
            conn = h5f.create_dataset("trl", shape=outShape, dtype=outdtype)
        
        for nf in range(nFreq):
            idx[freqidx] = nf
            dat = np.squeeze(trl_dat[idx])
            tmp = np.dot(dat.reshape(nChannel, nTaper), dat.reshape(nTaper, nChannel))/nTaper
            if pownorm:
                tdg = np.diag(tmp)        
                tmp /= np.sqrt(np.repeat(tdg.reshape(-1, 1), axis=1, repeats=nChannel) *
                               np.repeat(tdg.reshape(1, -1), axis=0, repeats=nChannel))
            conn[:, nf, :, :] = complexConversions[complex](tmp)
            if not inMemory:
                conn.flush()
                dat.flush()
                
    return conn

    
def _corr(blk, nChannel, nTaper, pownorm):
    
    tmp = da.dot(da.squeeze(blk).reshape(nChannel, nTaper), 
                 da.squeeze(blk).reshape(nTaper, nChannel))/nTaper
    if pownorm:
        tdg = da.diag(tmp)        
        tmp /= da.sqrt(da.repeat(tdg.reshape(-1, 1), axis=1, repeats=nChannel) *
                       da.repeat(tdg.reshape(1, -1), axis=0, repeats=nChannel))
    return complexConversions[complex](tmp).reshape(1, 1, nChannel, nChannel)


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
            
