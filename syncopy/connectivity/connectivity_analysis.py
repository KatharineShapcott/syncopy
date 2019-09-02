# -*- coding: utf-8 -*-
# 
# Syncopy connectivity analysis manager
# 
# Created: 2019-07-24 14:22:44
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-09-02 16:58:48>

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser, unwrap_cfg
from syncopy.datatype import ConnectivityData
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.connectivity.corr import ConnectivityCorr, ConnectivityCoh
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
def connectivityanalysis(data, method='coh', partchannel=None, complex="abs", 
                         removemean=True, bandwidth=None, 
                         keeptrials=True, out=None, **kwargs):
    """
    Coming soon...
    """
    
    # Ensure a valid computational method was selected
    avail_methods = ["coh", "corr", "powcorr", "cov", "csd"]
    # # FIXME: Supported methods in FieldTrip
    # avail_methods = ["amplcorr", "coh", "csd", "dtf", "granger", "pdc", "plv", 
    #                  "powcorr", "powcorr_ortho", "ppc", "psi", "wpli", "wpli_debiased", 
    #                  "wppc", "corr"]
    if method not in avail_methods:
        lgl = "'" + "or '".join(opt + "' " for opt in avail_methods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)
    
    # Method selection determines validity of input data
    supported_classes = {"coh": ["SpectralData"],
                         "corr": ["AnalogData"],
                         "cov": ["AnalogData"],
                        #  "corr": ["AnalogData", "SpectralData"],  # FIXME: add support for time-frequency data
                        #  "cov": ["AnalogData", "SpectralData"],  # FIXME: add support for time-frequency data
                         "csd": ["SpectralData"],
                         "powcorr": ["SpectralData"]}
    supported_dtypes = {"coh": "complex",
                        "corr": "float",
                        "csd": "complex",
                        "powcorr": "float"}
    dclass = data.__class__.__name__
    if dclass not in supported_classes[method]:
        msg = "Syncopy " + "or ".join(opt + " " for opt in supported_classes[method]) + "object"
        raise SPYTypeError(data, varname="data", expected=msg)
    if method in supported_dtypes.keys():
        if supported_dtypes[method] not in str(data.data.dtype):
            lgl = "{}-valued data".format(supported_dtypes[method])
            raise SPYValueError(legal=lgl, varname="data", actual=str(data.data.dtype))
    try:
        data_parser(data, varname="data", writable=None, empty=False)
    except Exception as exc:
        raise exc
        
    # Ensure output selection for complex-valued data makes sense
    if method in ["coh", "csd", "cov"]:
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
    if dclass == "SpectralData":
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
    log_dct = {"method": method, 
               "partchannel": partchannel, 
               "keeptrials": keeptrials}
    if method == "coh":
        connMethod = ConnectivityCoh(dimord, complex=complex)
        log_dct["complex"] = complex
    elif method == "corr":
        connMethod = ConnectivityCorr(dimord, complex="real", pownorm=True)
    elif method == "powcorr":
        connMethod = ConnectivityCorr(dimord, complex="real", pownorm=True, removemean=removemean)
        log_dct["removemean"] = removemean
    elif method == "amplcorr":
        connMethod = ConnectivityCorr(dimord, complex="real", pownorm=True, removemean=removemean)
        log_dct["removemean"] = removemean
    elif method == "cov":
        connMethod = ConnectivityCorr(dimord, complex=complex, pownorm=False)
        log_dct["complex"] = complex

    # Detect if dask client is running to set `parallel` keyword below accordingly
    use_dask = False
    if __dask__:
        try:
            dd.get_client()
            use_dask = True
        except ValueError:
            use_dask = False

    # Perform actual computation
    connMethod.initialize(data, chan_per_worker=kwargs.get("chan_per_worker"))
    connMethod.compute(data, out, parallel=use_dask, log_dict=log_dct)

    # Either return newly created output container or simply quit
    return out if new_out else None
