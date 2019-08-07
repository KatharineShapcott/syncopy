# -*- coding: utf-8 -*-
# 
# Example illustrating functionality of the connectivity module
# 
# Created: 2019-08-06 10:00:25
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-07 09:36:57>

# Builtin/3rd party package imports
import dask.distributed as dd

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)
import numpy as np

# Import SynCoPy
import syncopy as spy

# Import artificial data generator
from syncopy.tests.misc import generate_artifical_data

if __name__ == "__main__":

    # Create small `AnalogData` object for testing
    artdata = generate_artifical_data(nTrials=5, nChannels=16, equidistant=True, 
                                      inmemory=True)

    # Create uniform `cfg` struct holding analysis config
    cfg = spy.StructDict()
    cfg.method = "mtmfft"
    cfg.taper = "dpss"
    cfg.tapsmofrq = 9.3
    cfg.output = "abs"
    
    # Perform spectral analysis
    spec = spy.freqanalysis(artdata, cfg)
    
    sys.exit()
    
    # Take result of spectral analysis to compute all-to-all coherence
    conn = spy.connectivityanalysis(spec)
