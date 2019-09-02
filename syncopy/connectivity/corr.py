# -*- coding: utf-8 -*-
# 
# Syncopy correlation/coherence/covariance et al. compute kernels
# 
# Created: 2019-07-24 14:22:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-09-02 17:07:29>

# Builtin/3rd party package imports
import h5py
import numpy as np

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import AnalogData, SpectralData, ConnectivityData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.connectivity.connectivity_analysis import complexDTypes, complexConversions
from syncopy import __dask__
if __dask__:
    import dask.distributed as dd
    import dask.array as da


class ConnectivityCoh(ComputationalRoutine):
    
    computeFunction = staticmethod(coh)
    
    def process_metadata(self, data, out):
        
        out.channel1 = data.channel  # FIXME: this should become `data.channel[chanidx1]` at some point
        out.channel2 = data.channel  # FIXME: this should become `data.channel[chanidx2]` at some point
        out.freq = data.freq
        out.trialinfo = data .trialinfo


class ConnectivityCorr(ComputationalRoutine):
    
    computeFunction = staticmethod(corr)
    
    def process_metadata(self, data, out):
        
        out.channel1 = data.channel  # FIXME: this should become `data.channel[chanidx1]` at some point
        out.channel2 = data.channel  # FIXME: this should become `data.channel[chanidx2]` at some point
        if isinstance(out, SpectralData):
            out.freq = data.freq
        out.trialinfo = data .trialinfo
        

def corr(trl_dat, dimord, removemean=None, pownorm=True, complex="abs",
         noCompute=False, chunkShape=None):
    """
    Coming soon...
    """
    
    # Determine dimensional order of input
    if dimord != list(range(len(dimord))):
        shp = tuple([trl_dat.shape[dim] for dim in dimord])
    else:
        shp = trl_dat.shape
        
    # Set expected output shape depending on input (`AnalogData` or `SpectralData`)
    if len(shp) == 4:
        (_, nTaper, nFreq, nChannel) = shp
        outShape = (1, nFreq, nChannel, nChannel)
        outdtype = complexDTypes[complex]
        isSpectral = True
    else:
        (_, nChannel) = shp
        outShape = (1, nChannel, nChannel)
        outdtype = trl_dat.dtype
        isSpectral = False
        # if len(shp) == 3:
        #     dat = trl_dat.squeeze()  # only creates a view
        #     nuShape = outShape
        # else:
        #     dat = trl_dat
        #     nuShape = None
        # useNumpy = True
        
    # Get outta here if we're in the dry-run phase 
    if noCompute:
        return outShape, outdtype
    
    # The simple case: compute correlation or covariance of time-series
    if isSpectral:
        # average across tapers
    else:
        chanidx = dimord[1]
        if chanidx == 0:
            rowvar = True
        else:
            rowvar = False
        if pownorm:
            conn = np.corrcoef(dat, rowvar=rowvar)
        else:
            conn = np.cov(dat, rowvar=rowvar)
            
    return conn
    
    # The fun part...        
    else:

        # See if we can (further) parallelize across frequencies
        if __dask__:
            try:
                dd.get_client()
                parallel = False
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
                dat = np.squeeze(trl_dat[tuple(idx)]).reshape(nChannel, nTaper)
                tmp = np.dot(dat, dat.T)/nTaper
                if pownorm:
                    tdg = np.diag(tmp)
                    tmp /= np.sqrt(np.repeat(tdg.reshape(-1, 1), axis=1, repeats=nChannel) *
                                   np.repeat(tdg.reshape(1, -1), axis=0, repeats=nChannel))
                    import ipdb; ipdb.set_trace()
                conn[:, nf, :, :] = complexConversions[complex](tmp)
                if not inMemory:
                    conn.flush()
                    dat.flush()
                
    return conn

# def corr(trl_dat, dimord, pownorm=True, complex="abs", removemean=None,
#          noCompute=False, chunkShape=None):
#     """
#     pownorm   = flag that specifies whether normalisation with the
#                 product of the power should be performed (thus should
#                 be true when correlation/coherence is requested, and
#                 false when covariance or cross-spectral density is
#                 requested).
#     """
    
#     # Determine dimensional order of input
#     if dimord != list(range(len(dimord))):
#         shp = tuple([trl_dat.shape[dim] for dim in dimord])
#     else:
#         shp = trl_dat.shape
        
#     # Set expected output shape: if input comes from an `AnalogData` object, we
#     # simply use NumPy to compute correlation or covariance, otherwise things get nasty
#     if len(shp) == 4:
#         (_, nTaper, nFreq, nChannel) = shp
#         outShape = (1, nFreq, nChannel, nChannel)
#         outdtype = complexDTypes[complex]
#         useNumpy = False
#     else:
#         # Careful: if input is a dask-array chunk, `trl_dat` is 3-dimensional!
#         nChannel = shp[-1]
#         outShape = (1, nChannel, nChannel)
#         outdtype = trl_dat.dtype
#         if len(shp) == 3:
#             dat = trl_dat.squeeze()  # only creates a view
#             nuShape = outShape
#         else:
#             dat = trl_dat
#             nuShape = None
#         useNumpy = True
        
#     # Get outta here if we're in the dry-run phase 
#     if noCompute:
#         return outShape, outdtype
    
#     # The simple case: compute correlation or covariance of time-series
#     if useNumpy:
#         chanidx = dimord[AnalogData().dimord.index("channel")]
#         if chanidx == 0:
#             rowvar = True
#         else:
#             rowvar = False
#         if pownorm:
#             conn = np.corrcoef(dat, rowvar=rowvar).reshape(nuShape)
#         else:
#             conn = np.cov(dat, rowvar=rowvar).reshape(nuShape)
    
#     # The fun part...        
#     else:

#         # See if we can (further) parallelize across frequencies
#         if __dask__:
#             try:
#                 dd.get_client()
#                 parallel = False
#             except ValueError:
#                 parallel = False
            
#         # Get index of frequency dimension in input
#         freqidx = dimord[SpectralData().dimord.index("freq")] 
            
#         if parallel:
            
#             # Create a dask array chunked by frequency and map each taper-channel
#             # block onto `_corr` to compute channel x channel coherence/covariance
#             chunks = list(trl_dat.shape)
#             chunks[freqidx] = 1
#             dat = da.from_array(trl_dat, chunks=tuple(chunks))
            
#             conn = dat.map_blocks(_corr, nChannel, nTaper, pownorm, complex,
#                                   dtype=outdtype, 
#                                   chunks=(1, 1, nChannel, nChannel))
#             conn = conn.reshape(1, nFreq, nChannel, nChannel)
            
#             # Stacking gymnastics in case trials have different lengths (i.e., frequency-counts)
#             if nFreq < outShape[1]:
#                 conn = da.hstack([conn, 
#                                   da.zeros((1, outShape[1] - nFreq, nChannel, nChannel), 
#                                            dtype=conn.dtype)])
                
#         else:

#             # If trials don't fit into memory, `trl_dat` is already an HDF5 dataset -
#             # create a new HDF5 file parallel to `trl_dat`'s parent file and write results
#             # directly to it - flush after each frequency pass to not overflow memory
#             idx = [slice(None)] * len(dimord)
#             if inMemory:
#                 conn = np.full(chunkShape, np.nan, dtype=outdtype)
#             else:
#                 h5f = h5py.File(trl_dat.file.filename.replace(".h5", "_result.h5"))
#                 conn = h5f.create_dataset("trl", shape=outShape, dtype=outdtype)
                
#             # If trials don't fit into memory, flush after each frequency pass
#             idx = [slice(None)] * len(dimord)            
#             for nf in range(nFreq):
#                 idx[freqidx] = nf
#                 dat = np.squeeze(trl_dat[tuple(idx)]).reshape(nChannel, nTaper)
#                 tmp = np.dot(dat, dat.T)/nTaper
#                 if pownorm:
#                     tdg = np.diag(tmp)
#                     tmp /= np.sqrt(np.repeat(tdg.reshape(-1, 1), axis=1, repeats=nChannel) *
#                                    np.repeat(tdg.reshape(1, -1), axis=0, repeats=nChannel))
#                     import ipdb; ipdb.set_trace()
#                 conn[:, nf, :, :] = complexConversions[complex](tmp)
#                 if not inMemory:
#                     conn.flush()
#                     dat.flush()
                
#     return conn

