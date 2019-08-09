# -*- coding: utf-8 -*-
# 
# Syncopy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-09 16:21:38>

# Builtin/3rd party package imports
import sys
import numpy as np
import scipy.signal.windows as spwin
from tqdm import tqdm
import h5py
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)
from copy import copy
from numbers import Number

# Local imports
from syncopy.shared.parsers import (data_parser, scalar_parser, array_parser, 
                                    method_keyword_parser, unwrap_cfg)
from syncopy.shared import get_defaults
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import SpectralData, padding
import syncopy.specest.wavelets as spywave 
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy import __dask__
if __dask__:
    import dask
    import dask.array as da
    import dask.bag as db
    import dask.distributed as dd

# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "fourier": np.complex128,
                  "abs": np.float32}
spectralConversions = {"pow": lambda x: np.float32(x * np.conj(x)),
                       "fourier": lambda x: np.complex128(x),
                       "abs": lambda x: np.float32(np.absolute(x))}

__all__ = ["freqanalysis"]


@unwrap_cfg
def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, foi=None, pad='nextpow2', padtype='zero',
                 padlength=None, polyremoval=False, polyorder=None,
                 taper="hann", tapsmofrq=None, keeptapers=True,
                 wav="Morlet", toi=0.1, width=6,
                 out=None, **kwargs):
    """
    Explain taperopt...
    Explain default of toi (value b/w 0 and 1 indicating percentage of samplerate
    to use as stepsize)
    """

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(freqanalysis)
    lcls = locals()

    # List to keep track of keywords that were `None` in call but get overwritten
    NoneVals = []
    
    # Ensure a valid computational method was selected
    avail_methods = ["mtmfft", "wavelet"]
    if method not in avail_methods:
        lgl = "'" + "or '".join(opt + "' " for opt in avail_methods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Ensure a valid output format was selected
    options = ["pow", "fourier", "abs"]
    if output not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # Patch `output` keyword to not collide w/dask's ``blockwise`` output
    defaults["output_fmt"] = defaults.pop("output")
    
    # We need to define `output_fmt` for `locals()` call in `method_keyword_parser`!
    output_fmt = output

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers", "polyremoval"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")

    # Ensure padding selection makes sense (just leverage `padding`'s error checking)
    if pad is not None:
        try:
            padding(data.trials[0], padtype, pad=pad, padlength=padlength,
                    prepadlength=True)
        except Exception as exc:
            raise exc

    # For vetting `toi` and `foi`: get timing information of input object
    timing = np.array([np.array([-data.t0[k], end - start - data.t0[k]])/data.samplerate
                       for k, (start, end) in enumerate(data.sampleinfo)])

    # Construct array of maximally attainable frequency band and set/align `foi`
    minSampleNum = np.diff(data.sampleinfo).min()
    if pad:
        minSamplePos = np.diff(data.sampleinfo).argmin()
        minSampleNum = padding(data.trials[minSamplePos], padtype, pad=pad,
                               padlength=padlength, prepadlength=True).shape[timeAxis]
    minTrialLength = minSampleNum/data.samplerate
    nFreq = int(np.floor(minSampleNum / 2) + 1)
    freqs = np.linspace(0, data.samplerate/2, nFreq)

    # Match desired frequencies as close as possible to actually attainable freqs
    if foi is not None:
        try:
            array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                         lims=[1/minTrialLength, data.samplerate/2], dims=(None,))
        except Exception as exc:
            raise exc
        foi = np.array(foi)
        foi.sort()
        foi = foi[foi <= freqs.max()]
        foi = foi[foi >= freqs.min()]
        foi = freqs[np.unique(np.searchsorted(freqs, foi, side="right") - 1)]
    else:
        foi = freqs
        NoneVals.append("foi")

    # FIXME: implement detrending
    if polyremoval is True or polyorder is not None:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Check detrending options for consistency
    if polyremoval:
        try:
            scalar_parser(polyorder, varname="polyorder", lims=[0, 8], ntype="int_like")
        except Exception as exc:
            raise exc
    else:
        if polyorder != defaults["polyorder"]:
            print("<freqanalysis> WARNING: `polyorder` keyword will be ignored " +
                  "since `polyremoval` is `False`!")

    # Ensure consistency of method-specific options
    if method == "mtmfft":

        options = ["hann", "dpss"]
        if taper not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="taper", actual=taper)
        taper = getattr(spwin, taper)

        # Advanced usage: see if `taperopt` was provided - if not, leave it empty
        taperopt = kwargs.get("taperopt", {})
        if not isinstance(taperopt, dict):
            raise SPYTypeError(taperopt, varname="taperopt", expected="dictionary")

        # Set/get `tapsmofrq` if we're working w/Slepian tapers
        if taper.__name__ == "dpss":

            # Try to derive "sane" settings by using 3/4 octave smoothing of highest `foi`
            # following Hill et al. "Oscillatory Synchronization in Large-Scale
            # Cortical Networks Predicts Perception", Neuron, 2011
            if tapsmofrq is None:
                foimax = foi.max()
                tapsmofrq = (foimax * 2**(3/4/2) - foimax * 2**(-3/4/2)) / 2
                NoneVals.append("tapsmofrq")
            else:
                try:
                    scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[1, np.inf])
                except Exception as exc:
                    raise exc

        # Warn the user in case `tapsmofrq` has no effect
        if tapsmofrq is not None and taper.__name__ != "dpss":
            print("<freqanalysis> WARNING: `tapsmofrq` is only used if `taper` is `dpss`!")

    elif method == "wavelet":
        options = ["Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat"]
        if wav not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="wav", actual=wav)
        wav = getattr(spywave, wav)

        if isinstance(toi, Number):
            try:
                scalar_parser(toi, varname="toi", lims=[0, 1])
            except Exception as exc:
                raise exc
        else:
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[timing.min(), timing.max()], dims=(None,))
            except Exception as exc:
                raise exc
            toi = np.array(toi)
            toi.sort()

        if foi is None:
            foi = 1 / _get_optimal_wavelet_scales(minTrialLength,
                                                  1/data.samplerate,
                                                  dj=0.25)

        # FIXME: width setting depends on chosen wavelet
        if width is not None:
            try:
                scalar_parser(width, varname="width", lims=[1, np.inf])
            except Exception as exc:
                raise exc

    # Construct method-specific dict of input keywords and a dict for logging
    mth_input, log_dct = method_keyword_parser(method, list(avail_methods))
    mth_input["keeptrials"] = keeptrials
    for key in NoneVals:
        log_dct[key] = None
        
    # If provided, make sure output object is appropriate
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True, empty=True,
                        dataclass="SpectralData",
                        dimord=SpectralData().dimord)
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = SpectralData()
        new_out = True

    # Construct dict of classes of available methods
    methods = {
        "mtmfft": MultiTaperFFT(1/data.samplerate, timeAxis, **mth_input),
        "wavelet": WaveletTransform(1/data.samplerate, timeAxis, foi, **mth_input)
    }

    # Detect if dask client is running and set `parallel` keyword accordingly
    use_dask = False
    if __dask__:
        try:
            dd.get_client()
            use_dask = True
        except ValueError:
            use_dask = False

    # Perform actual computation
    specestMethod = methods[method]
    specestMethod.initialize(data)
    specestMethod.compute(data, out, parallel=use_dask, log_dict=log_dct)

    # Either return newly created output object or simply quit
    return out if new_out else None


# Local workhorse that performs the computational heavy lifting
def mtmfft(trl_dat, dt, timeAxis,
           taper=spwin.hann, taperopt={}, tapsmofrq=None,
           pad="nextpow2", padtype="zero", padlength=None, foi=None,
           keeptapers=True, polyorder=None, output_fmt="pow",
           noCompute=False, chunkShape=None, inMemory=True):

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat
    dat = dat.squeeze()

    # Padding (updates no. of samples)
    if pad is not None:
        dat = padding(dat, padtype, pad=pad, padlength=padlength, prepadlength=True)
    nSamples = dat.shape[0]
    nChannels = dat.shape[1]

    # Construct at least 1 and max. 50 taper(s)
    if taper == spwin.dpss and (not taperopt):
        nTaper = int(max(2, min(50, np.floor(tapsmofrq * nSamples * dt))))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}
    win = np.atleast_2d(taper(nSamples, **taperopt))
    nTaper = win.shape[0]

    # Determine frequency band and shape of output (time=1 x taper x freq x channel)
    nFreq = int(np.floor(nSamples / 2) + 1)
    fidx = slice(None)
    if foi is not None:
        freqs = np.linspace(0, 1 / (2 * dt), nFreq)
        foi = foi[foi <= freqs.max()]
        foi = foi[foi >= freqs.min()]
        fidx = np.searchsorted(freqs, foi, side="right") - 1
        nFreq = fidx.size
    outShape = (1, max(1, nTaper * keeptapers), nFreq, nChannels)

    # For initialization of computational routine, just return output shape and dtype
    if noCompute:
        return outShape, spectralDTypes[output_fmt]

    # In case tapers aren't kept allocate `spec` "too big" and average afterwards
    if not keeptapers:
        shp = list(chunkShape)
        shp[1] = nTaper
        chunkShape = tuple(shp)
    spec = np.full(chunkShape, np.nan, dtype=spectralDTypes[output_fmt])
    fill_idx = tuple([slice(None, dim) for dim in outShape[2:]])

    # Actual computation
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1)).T
        spec[(0, taperIdx,) + fill_idx] = spectralConversions[output_fmt](np.fft.rfft(dat * taper, axis=0)[fidx, :])

    # Average across tapers if wanted
    if not keeptapers:
        return spec.mean(axis=1, keepdims=True)
    else:
        return spec


class MultiTaperFFT(ComputationalRoutine):

    computeFunction = staticmethod(mtmfft)

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

def wavelet(trl_dat, dt, timeAxis, foi,
            toi=0.1, polyorder=None, wav=spywave.Morlet,
            width=6, output_fmt="pow",
            noCompute=False, chunkShape=None):
    """ dat = samples x channel
    """

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T.squeeze()       # does not copy but creates a view of `trl_dat`
    else:
        dat = trl_dat
    nSamples = dat.shape[0]
    nChannels = dat.shape[1]

    # Initialize wavelet
    

    # Get time-stepping or explicit time-points of interest
    if isinstance(toi, Number):
        toi /= dt
        tsize = int(np.floor(nSamples / toi))
    else:
        tsize = toi.size

    # Output shape: time x taper=1 x freq x channel
    import ipdb; ipdb.set_trace()
    scales = wav.scale_from_period(1/foi)
    outShape = (tsize,
                1,
                len(scales),
                nChannels)

    # For initialization of computational routine, just return output shape and dtype
    if noCompute:
        return outShape, spectralDTypes[output_fmt]

    # Actual computation: ``cwt`` returns `(len(scales),) + dat.shape`
    transformed = spywave.cwt(dat, axis=0, wavelet=wav, widths=scales, dt=dt)
    transformed = transformed[:, 0:-1:tsize, :, np.newaxis].transpose([1, 3, 0, 2])

    return spectralConversions[output_fmt](transformed)


class WaveletTransform(ComputationalRoutine):

    computeFunction = staticmethod(wavelet)

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def compute_parallel(self, data, out):

        # Point to trials on disk by using delayed **static** method calls
        lazy_trial = dask.delayed(data._copy_trial, traverse=False)
        lazy_trls = [lazy_trial(trialno,
                                data._filename,
                                data.dimord,
                                data.sampleinfo,
                                data.hdr)
                     for trialno in range(data.sampleinfo.shape[0])]

        # Stack trials along new (3rd) axis inserted on the left
        trl_block = da.stack([da.from_delayed(trl, shape=data._shapes[sk],
                                              dtype=data.data.dtype)
                              for sk, trl in enumerate(lazy_trls)])

        # Use `map_blocks` to compute spectra for each trial in the
        # constructed dask array
        specs = trl_block.map_blocks(self.computeFunction,
                                     *self.argv, **self.cfg,
                                     dtype="complex",
                                     chunks=self.cfg["chunkShape"],
                                     new_axis=[0])
        specs = specs.reshape(self.outputShape)

        # Average across trials if wanted
        if not self.keeptrials:
            specs = specs.mean(axis=0)

        return specs

    # def initialize(self, data):
    #     timeAxis = data.dimord.index("time")
    #     minTrialLength = np.array(data._shapes)[:, timeAxis].min()
    #     if self.cfg["foi"] is None:
    #         self.cfg["foi"] = 1 / _get_optimal_wavelet_scales(minTrialLength,
    #                                                           1 / data.samplerate,
    #                                                           dj=0.25)
    # 
    #     if self.cfg["toi"] is None:
    #         asdf
    #         1/foi[-1] * w0
    # 
    #     dryRunKwargs = copy(self.cfg)
    #     dryRunKwargs["noCompute"] = True
    #     self.chunkShape, self.dtype = self.computeFunction(data.trials[0],
    #                                                         *self.argv, **dryRunKwargs)

    def preallocate_output(self, data, out):
        totalTriallength = np.int(np.sum(np.floor(np.array(data._shapes)[:, 0] /
                                                  self.cfg["stepsize"])))

        result = open_memmap(out._filename,
                             shape=(totalTriallength,) + self.chunkShape[1:],
                             dtype=self.dtype,
                             mode="w+")
        del result

    def compute_sequential(self, data, out):
        outIndex = 0
        for idx, trial in enumerate(tqdm(data.trials,
                                         desc="Computing Wavelet spectrum...")):
            tmp = self.computeFunction(trial, *self.argv, **self.cfg)
            selector = slice(outIndex, outIndex + tmp.shape[0])
            res = open_memmap(out._filename, mode="r+")[selector, :, :, :]
            res[...] = tmp
            del res
            data.clear()

    def compute_with_dask(self, data, out):
        raise NotImplementedError("Dask computation of wavelet transform is not yet implemented")

    def process_metadata(self, data, out):
        out.data = open_memmap(out._filename, mode="r+")
        # We can't simply use ``redefinetrial`` here, prep things by hand
        out.sampleinfo = np.floor(data.sampleinfo / self.cfg["stepsize"]).astype(np.int)
        out.trialinfo = np.array(data.trialinfo)
        out._t0 = data._t0 / self.cfg["stepsize"]

        # Attach meta-data
        out.samplerate = data.samplerate / self.cfg["stepsize"]
        out.channel = np.array(data.channel)
        out.freq = self.cfg["freqoi"]
        return out


WaveletTransform.computeMethods = {"dask": WaveletTransform.compute_with_dask,
                                   "sequential": WaveletTransform.compute_sequential}


def _get_optimal_wavelet_scales(nSamples, dt, dj=0.25, s0=1):
    """Form a set of scales to use in the wavelet transform.

    For non-orthogonal wavelet analysis, one can use an
    arbitrary set of scales.

    It is convenient to write the scales as fractional powers of
    two:

        s_j = s_0 * 2 ** (j * dj), j = 0, 1, ..., J

        J = (1 / dj) * log2(N * dt / s_0)

    s0 - smallest resolvable scale
    J - largest scale

    choose s0 so that the equivalent Fourier period is 2 * dt.

    The choice of dj depends on the width in spectral space of
    the wavelet function. For the Morlet, dj=0.5 is the largest
    that still adequately samples scale. Smaller dj gives finer
    scale resolution.
    """
    s0 = 2 * dt
    # Largest scale
    J = int((1 / dj) * np.log2(nSamples * dt / s0))
    return s0 * 2 ** (dj * np.arange(0, J + 1))


def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n
