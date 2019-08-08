# -*- coding: utf-8 -*-
# 
# Syncopy connectivity analysis methods
# 
# Created: 2019-07-24 14:22:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-08 12:26:32>

# Builtin/3rd party package imports
import os
import sys
import h5py
import numpy as np

# Local imports
from syncopy.shared.parsers import (data_parser, scalar_parser, array_parser, 
                                    method_keyword_parser, unwrap_cfg)
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import AnalogData, SpectralData, ConnectivityData
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
def connectivityanalysis(data, method='coh', partchannel=None, complex="abs", 
                         removemean=True, bandwidth=None, 
                         out=None):
    """
    Coming soon...
    """
    
    # Ensure a valid computational method was selected
    avail_methods = ["coh", "corr", "cov", "csd"]
    # # FIXME: FT supported methods:
    # avail_methods = ["amplcorr", "coh", "csd", "dtf", "granger", "pdc", "plv", 
    #                  "powcorr", "powcorr_ortho", "ppc", "psi", "wpli", "wpli_debiased", 
    #                  "wppc", "corr"]
    if method not in avail_methods:
        lgl = "'" + "or '".join(opt + "' " for opt in avail_methods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)
    
    # Method selection determines validity of input datatype
    if method in ["corr", "cov"]:
        if data.__class__.__name__ not in ["AnalogData", "SpectralData"]:
            msg = "Syncopy AnalogData or SpectralData object"
            raise SPYTypeError(data, varname="data", expected=msg)
        dclass = data.__class__.__name__
    else:
        dclass = "SpectralData"
    try:
        data_parser(data, varname="data", dataclass=dclass, writable=None, empty=False)
    except Exception as exc:
        raise exc
        
    # Ensure output selection for complex-valued data makes sense
    if dclass == "SpectralData":
        options = complexConversions.keys()
        if complex not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="complex", actual=complex)

    # Get positional indices of dimensions in `data` relative to class defaults
    dimord = [data.dimord.index(dim) for dim in data.__class__().dimord]
    
    # Parsing of method-specific input parameters
    if method in ["coh", "csd", "plv"]:
        if partchannel is not None:
            raise NotImplementedError("Partial coherence not yet implemented")
            try:
                array_parser(partchannel, varname="partchannel", ntype="str", 
                            dims=data.channel.size)
            except Exception as exc:
                raise exc
            if np.any([chan not in data.channel for chan in partchannel]):
                lgl = "all channels to be partialized out to be present in input object"
                raise SPYValueError(legal=lgl, varname="partchannel")
            
    elif method in ["powcorr", "amplcorr"]:
        if not isinstance(removemean, bool):
            raise SPYTypeError(removemean, varname="removemean", expected="bool")
        
    elif method == "psi":
        try:
            scalar_parser(bandwidth, varname="bandwidth", lims=[0, data.freq.max()/2])
        except Exception as exc:
            raise exc
        
    # If provided, make sure output object is appropriate for type of input 
    dims = ["trial", "channel", "channel"]
    if isinstance(data, SpectralData):
        dims.insert(1, "freq")
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True, empty=True,
                        dataclass="ConnectivityData",
                        dimord=dims)
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = ConnectivityData(dimord=dims)
        new_out = True
        
    # Select appropriate compute kernel for requested method
    avail_kernels = ["corr"]
    if method in ["coh", "corr", "cov", "csd", "plv", "amplcorr", "powcorr"]:
        
        # Set power-normalization based on chosen method
        if method in ["cov", "csd"]:
            pownorm = False
        else:
            pownorm = True
            
        # Construct method-specific dict of input keywords and a dict for logging
        mth_input, log_dct = method_keyword_parser("corr", avail_kernels)
        connMethod = ConnectivityCorr(dimord, **mth_input)

    # Detect if dask client is running to set `parallel` keyword below accordingly
    if __dask__:
        try:
            dd.get_client()
            use_dask = True
        except ValueError:
            use_dask = False
            
    # Perform actual computation
    connMethod.initialize(data)
    connMethod.compute(data, out, parallel=use_dask, log_dict=log_dct)
    
    # Either return newly created output container or simply quit
    return out if new_out else None


def corr(trl_dat, dimord, pownorm=True, complex="abs",
         noCompute=False, chunkShape=None, inMemory=True):
    """
    pownorm   = flag that specifies whether normalisation with the
                product of the power should be performed (thus should
                be true when correlation/coherence is requested, and
                false when covariance or cross-spectral density is
                requested).
    """
    
    # Determine dimensional order of input
    if dimord != list(range(len(dimord))):
        shp = tuple([trl_dat.shape[dim] for dim in dimord])
    else:
        shp = trl_dat.shape
    
    # Set expected output shape: if input comes from an `AnalogData` object, we
    # simply use NumPy to compute correlation or covariance, otherwise things get nasty
    if len(shp) == 4:
        (_, nTaper, nFreq, nChannel) = shp
        outShape = (1, nFreq, nChannel, nChannel)
        outdtype = complexDTypes[complex]
        useNumpy = False
    else:
        (_, nChannel) = shp
        outShape = (1, nChannel, nChannel)
        outdtype = trl_dat.dtype
        useNumpy = True
        
    # Get outta here if we're in the dry-run phase 
    if noCompute:
        return outShape, outdtype
    
    # The simple case: compute correlation or covariance of time-series
    if useNumpy:
        chanidx = dimord[AnalogData().dimord.index("channel")]
        if chanidx == 0:
            rowvar = True
        else:
            rowvar = False
        if pownorm:
            conn = np.corrcoef(trl_dat, rowvar=rowvar)
        else:
            conn = np.cov(trl_dat, rowvar=rowvar)
    
    # The fun part...        
    else:

        # See if we can (further) parallelize across frequencies
        if __dask__:
            try:
                dd.get_client()
                # if nFreq * nChannel 
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
            
            conn = dat.map_blocks(_corr, nChannel, nTaper, pownorm, complex,
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
                
            # If trials don't fit into memory, flush after each frequency pass
            idx = [slice(None)] * len(dimord)            
            for nf in range(nFreq):
                idx[freqidx] = nf
                dat = np.squeeze(trl_dat[tuple(idx)])
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

    
def _corr(blk, nChannel, nTaper, pownorm, complex):
    
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
        
        out.channel1 = data.channel  # FIXME: this should become `data.channel[chanidx1]` at some point
        out.channel2 = data.channel  # FIXME: this should become `data.channel[chanidx2]` at some point
        if isinstance(data, SpectralData):
            out.freq = data.freq
        out.trialinfo = data .trialinfo
