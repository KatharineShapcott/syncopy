# -*- coding: utf-8 -*-
# 
# Example illustrating processing *large* datasets in Syncopy
# 
# Created: 2019-07-26 10:20:58
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-26 10:50:32>

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

    # Either load already prepared large (ca. 36 GB) dataset or create one
    # NOTE: creating from scractch takes *a while*!
    container = "/mnt/hpx/it/dev/syncopy-testdata/BigData"
    try:
        big = spy.load(container)
    except:
        big = generate_artifical_data(nTrials=100, nChannels=4096, 
                                      samplerate=8000, inmemory=False) 
