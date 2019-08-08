# -*- coding: utf-8 -*-
# 
# Syncopy container class for statistical data
# 
# Created: 2019-07-25 13:57:57
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-08 11:17:04>

# Builtin/3rd party package imports
import numpy as np
from abc import ABC

# Local imports
from .base_data import BaseData
from syncopy.shared.parsers import array_parser
from syncopy.shared.errors import SPYValueError

__all__ = ["ConnectivityData"]


class ConnectivityData(BaseData, ABC):
    
    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        idx = [slice(None)] * len(self.dimord)
        if "trial" not in self.dimord and trialno > 0:
            lgl = "trial no. 0 for single-trial dataset"
            raise SPYValueError(legal=lgl, varname="trial-selector", actual=str(trialno))
        else:    
            idx[self.dimord.index("trial")] = trialno
        return self._data[tuple(idx)]

    # dummy to not upset abstract method    
    def selectdata(self):
        pass
    
    @property
    def channel1(self):
        """ :class:`numpy.ndarray` : list of channel names """
        return self._channel1

    @channel1.setter
    def channel1(self, chan):
        if self.data is None:
            print("SyNCoPy core - channel1: Cannot assign `channels` without data. " +
                  "Please assing data first")
            return
        nchan = self.data.shape[self.dimord.index("channel")]
        try:
            array_parser(chan, varname="channel1", ntype="str", dims=(nchan,))
        except Exception as exc:
            raise exc
        self._channel1 = np.array(chan)
        
    @property
    def channel2(self):
        """ :class:`numpy.ndarray` : list of channel names """
        return self._channel2

    @channel2.setter
    def channel2(self, chan):
        if self.data is None:
            print("SyNCoPy core - channel2: Cannot assign `channels` without data. " +
                  "Please assing data first")
            return
        dims = [idx for idx, val in enumerate(self.dimord) if val == 'channel']     
        nchan = self.data.shape[dims[1]]
        try:
            array_parser(chan, varname="channel2", ntype="str", dims=(nchan,))
        except Exception as exc:
            raise exc
        self._channel2 = np.array(chan)
        
    @property
    def freq(self):
        """:class:`numpy.ndarray`: frequency axis in Hz """
        return self._freq

    @freq.setter
    def freq(self, freq):
        if "freq" not in self.dimord:
            print("SyNCoPy core - freq: Object does not have frequency dimension.")
            return
        if self.data is None:
            print("SyNCoPy core - freq: Cannot assign `freq` without data. "+\
                  "Please assing data first")
            return
        nfreq = self.data.shape[self.dimord.index("freq")]
        try:
            array_parser(freq, varname="freq", dims=(nfreq,), hasnan=False, hasinf=False)
        except Exception as exc:
            raise exc
        self._freq = np.array(freq)
    
    @property
    def sampleinfo(self):
        """undefined for :class:`syncopy.ConnectivityData` objects"""
        return None
    
    @sampleinfo.setter
    def sampleinfo(self, sinfo):
        print("SyNCoPy core - sampleinfo: ConnectivityData objects do not support "+\
              "sampleinfo assignments")
        return
    
    @property
    def trialinfo(self):
        """nTrials x M :class:`numpy.ndarray` with numeric information about each trial

        Each trial can have M properties (condition, original trial no., ...) coded by 
        """
        return self._trialinfo

    @trialinfo.setter
    def trialinfo(self, trl):
        if self.data is None:
            print("SyNCoPy core - trialinfo: Cannot assign `trialinfo` without data. "+\
                  "Please assing data first")
            return
        if "trial" not in self.dimord:
            trialdim = 1
        else:
            trialdim = self.data.shape[self.dimord.index("trial")]
        try:
            array_parser(trl, varname="trialinfo", dims=(trialdim, None))
        except Exception as exc:
            raise exc
        self._trialinfo = np.array(trl)
        
        
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 channel1="first_channel_dim",
                 channel2="second_channel_dim",
                 mode="w",
                 dimord=["trial", "freq", "channel", "channel"]):
        
        # The one thing we check right here and now
        if dimord.count("channel") != 2:
            lgl = "channel x channel specification"
            act = "'" + "' x '".join(str(dim) for dim in dimord) + "'"
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)
        expected = ["trial", "freq", "channel", "channel"]
        if not set(dimord).issubset(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)
        
        # For base correlation/covariance objects, ensure `_freq` is set
        self._freq = None

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         channel1=channel1,
                         channel2=channel2,
                         mode=mode,
                         dimord=dimord)

