# -*- coding: utf-8 -*-
# 
# Syncopy container class for statistical data
# 
# Created: 2019-07-25 13:57:57
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-26 16:42:52>

# Builtin/3rd party package imports
from abc import ABC

# Local imports
from .base_data import BaseData
from syncopy.shared.errors import SPYValueError

__all__ = ["ConnectivityData"]


class ConnectivityData(BaseData, ABC):
    
    # Helper function that grabs a single trial
    def _get_trial(self, trialno):
        idx = [slice(None)] * len(self.dimord)
        idx[self.dimord.index("time")] = slice(int(self.sampleinfo[trialno, 0]), 
                                               int(self.sampleinfo[trialno, 1]))
        return self._data[tuple(idx)]

    # dummy to not upset abstract method    
    def selectdata(self):
        pass
    
    def __init__(self,
                 data=None,
                 filename=None,
                 trialdefinition=None,
                 channel="channel",
                 mode="w",
                 dimord=["time", "frequency", "channel", "channel"]):

        # The one thing we check right here and now
        expected = ["time", "frequency", "channel", "channel"]
        if not set(dimord).issubset(expected):
            base = "dimensional labels {}"
            lgl = base.format("'" + "' x '".join(str(dim) for dim in expected) + "'")
            act = base.format("'" + "' x '".join(str(dim) for dim in dimord) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # Hard constraint: required no. of data-dimensions
        self._ndim = 4

        # Call parent initializer
        super().__init__(data=data,
                         filename=filename,
                         trialdefinition=trialdefinition,
                         channel=channel,
                         mode=mode,
                         dimord=dimord)

