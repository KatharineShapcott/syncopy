# -*- coding: utf-8 -*-
# 
# Example illustrating how to process *large* datasets in Syncopy
# 
# Created: 2019-07-26 10:20:58
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-12 14:21:08>

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

    # Set up parameters for large (high-res) dataset with relatively small (~375MB)
    # trials
    nTrials = 100
    nChannels = 4096
    samplerate = 8000

    # Either load already prepared large (ca. 36 GB) dataset or create one
    # NOTE: creating from scractch takes *a while*!
    container = "/mnt/hpx/it/dev/syncopy-testdata/BigData"
    try:
        big = spy.load(container)
    except:
        big = generate_artifical_data(nTrials=nTrials, nChannels=nChannels, 
                                      samplerate=samplerate, inmemory=False) 
        
    # Launcbh a SLURM worker swarm to handle this data set
    client = spy.esi_cluster_setup(partition="DEV", mem_per_job="3GB", n_jobs=2)
    
    # sys.exit()
    
    # Start by computing channel-by-channel correlations
    cfg = spy.StructDict()
    cfg.method = "corr"
    corr = spy.connectivityanalysis(big, cfg)
    
    # spec = spy.freqanalysis(big)