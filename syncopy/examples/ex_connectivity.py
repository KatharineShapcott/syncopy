# -*- coding: utf-8 -*-
# 
# Example illustrating functionality of the connectivity module
# 
# Created: 2019-08-06 10:00:25
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-12 12:40:34>

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
    nTrials = 5
    artdata = generate_artifical_data(nTrials=nTrials, nChannels=16, equidistant=True, 
                                      inmemory=True)
    
    # Create uniform `cfg` struct holding analysis config
    cfg = spy.StructDict()
    cfg.method = "mtmfft"
    cfg.taper = "dpss"
    cfg.tapsmofrq = 9.3
    cfg.output = "abs"
    
    client = dd.Client()
    conn = spy.connectivityanalysis(artdata, method="corr")
    # spy.connectivityanalysis(ff, method="corr")
   
    sys.exit()
    
    # Perform spectral analysis
    spec = spy.freqanalysis(artdata, cfg)
    
    # Take result of spectral analysis to compute all-to-all coherence
    cfg = spy.StructDict()
    cfg.method = "coh"
    cfg.complex = "abs"
    conn = spy.connectivityanalysis(spec, cfg)
    
    # # test coh, corr, cov, csd
    # client = spy.esi_cluster_setup(partition="DEV", mem_per_job="4GB", workers_per_job=1, n_jobs=nTrials)
    # res = spy.connectivityanalysis(spec, cfg)
    
    # conn_corr = spy.connectivityanalysis(artdata, method="corr")
    # conn_cov = spy.connectivityanalysis(artdata, method="cov")