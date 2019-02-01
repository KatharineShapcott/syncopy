# dframe_playground.py - Script to test usage of Pandas dataframes w/ our data
# 
# Created: Januar 25 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-01 18:16:41>

# Builtin/3rd party package imports
import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import numpy as np
import os
import sys
import pandas as pd
from memory_profiler import memory_usage

# Add spykewave package to Python search path
spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

# Import Spykewave
import spykewave as sw

# Binary flags controlling program flow
slurmComputation = False        # turn SLURM usage on/off and

# %% -------------------- Set up parallel environment --------------------
if slurmComputation:
    cluster = SLURMCluster(processes=8,
                           cores=8,
                           memory="48GB",                                              
                           queue="DEV")
    cluster.start_workers(1)    
    print("Waiting for workers to start")    
    while len(cluster.workers) == 0:
        time.sleep(0.5)
    client = Client(cluster)
    print(client)


# %% -------------------- Define location of test data --------------------
# datadir = "/mnt/hpx/it/dev/SpykeWave/testdata/"
datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
          + os.sep + "testdata" + os.sep
basename = "MT_RFmapping_session-168a1"


# %% -------------------- Define trial from photodiode onsets --------------------
pdFile = os.path.join(datadir, basename + ".dpd")
pdData = sw.BaseData(pdFile, filetype="esi")

# Helper functions to convert sample-no to actual time-code
def time2sample(t, dt=0.001):
    return (t/dt).astype(dtype=int)
def sample2time(s, dt=0.001):
    return s*dt

# Trials start 250 ms before stimulus onset
pdOnset = np.array(pdData._segments[:,:])
iStart = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1], 
                                 dt=pdData.hdr["tSample"]/1E9) - 0.25,
                     dt=0.001)
iEnd = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1],
                               dt=pdData.hdr["tSample"]/1E9) + 0.5,
                   dt=0.001)

# Construct trial definition matrix
intervals = np.stack((iStart,iEnd, np.tile(250, [iStart.size]).T), axis=1)

# Remove very short trials
intervals = intervals[intervals[:,1]-intervals[:,0] > 500]

# Read actual raw data from disk
dataFiles = [os.path.join(datadir, basename + ext) for ext in ["_xWav.lfp", "_xWav.mua"]]

# Read header from one file to get dimensional info (and not unfairly bias memory usage)
hdr = sw.read_binary_esi_header(dataFiles[0])
label = ["channel" + str(k) for k in range(1, 2*int(hdr["M"]) + 1)]
dsets = []
for dfile in dataFiles:
    dsets.append(np.memmap(dfile, offset=int(hdr["length"]),
                           mode="r", dtype=hdr["dtype"],
                           shape=(hdr["N"], hdr["M"])))

# Print reserved memory
print("Memory usage before anything was done: ", memory_usage()[0])

# Create `BaseData` instance and measure its memory footprint
data = sw.BaseData(dataFiles, trialdefinition=intervals, filetype="esi")
print("Memory usage after BaseData instantiation: ", memory_usage()[0])

# Create two dataframes whose contents is mem-mapped on disk
dframe1 = pd.DataFrame(dsets[0], columns=label[:int(hdr["M"])])
dframe2 = pd.DataFrame(dsets[1], columns=label[int(hdr["M"]):])
print("Memory usage after allocation of two dataframes: ", memory_usage()[0])

# Join memory-mapped dataframes (causes all memory maps to be loaded)
bigframe = pd.concat([dframe1, dframe2])
print("Memory usage after joining dataframes: ", memory_usage()[0])

# Delete the big 'one
del bigframe
print("Memory usage after deleting joined dataframe: ", memory_usage()[0])

# Try to leverage some delayed magic to avoid explicit loading of memmaps
delayed_concat = dask.delayed(pd.concat)
daskframe = dd.from_delayed(delayed_concat([df for df in [dframe1, dframe2]]))
print("Memory usage after delayed join: ", memory_usage()[0])
